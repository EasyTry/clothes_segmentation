import io
import numpy as np
import cv2
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from Graphonomy.networks import deeplab_xception_transfer, graph
from Graphonomy.dataloaders import custom_transforms as tr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

class ClothesSegmentator:
    def __init__(self):
        checkpoint_dir = './pretrained_weights/'
        self.model_restore = checkpoint_dir + 'inference.pth'
        self.use_gpu = True
        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                            hidden_layers=128,
                                                                            source_classes=7, )
        x = torch.load(self.model_restore)
        self.net.load_source_model(x)
        self.net.cuda()
        
    def read_img(self, image):
      _img = Image.fromarray(image).convert('RGB')  # return is RGB pic
      return _img

    def img_transform(self, img, transform=None):
        sample = {'image': img, 'label': 0}
        sample = transform(sample)
        return sample

    def flip(self, x, dim):
      indices = [slice(None)] * x.dim()
      indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                  dtype=torch.long, device=x.device)
      return x[tuple(indices)]

    def flip_cihp(self, tail_list):
      '''
      :param tail_list: tail_list size is 1 x n_class x h x w
      :return:
      '''
      # tail_list = tail_list[0]
      tail_list_rev = [None] * 20
      for xx in range(14):
          tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
      tail_list_rev[14] = tail_list[15].unsqueeze(0)
      tail_list_rev[15] = tail_list[14].unsqueeze(0)
      tail_list_rev[16] = tail_list[17].unsqueeze(0)
      tail_list_rev[17] = tail_list[16].unsqueeze(0)
      tail_list_rev[18] = tail_list[19].unsqueeze(0)
      tail_list_rev[19] = tail_list[18].unsqueeze(0)
      return torch.cat(tail_list_rev,dim=0)

    def __call__(self, data):
        
        image = np.load(io.BytesIO(data))
        
        original_h, original_w = image.shape[:2]
        small_image = cv2.resize(image.astype(np.uint8), (512, 512))

        # adj
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

        # multi-scale
        scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
        img = self.read_img(small_image)
        testloader_list = []
        testloader_flip_list = []
        for pv in scale_list:
            composed_transforms_ts = transforms.Compose([
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            testloader_list.append(self.img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(self.img_transform(img, composed_transforms_ts_flip))
        # print(testloader_list)
        
        # One testing epoch
        self.net.eval()
        # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

        for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            inputs = torch.cat((inputs, inputs_f), dim=0)
            if iii == 0:
                _, _, h, w = inputs.size()
            # assert inputs.size() == inputs_f.size()

            # Forward pass of the mini-batch
            inputs = Variable(inputs, requires_grad=False)

            with torch.no_grad():
                if self.use_gpu >= 0:
                    inputs = inputs.cuda()
                # outputs = net.forward(inputs)
                outputs = self.net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[0] + self.flip(self.flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii > 0:
                    outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()

        predictions = torch.max(outputs_final, 1)[1]
        small_seg = predictions.cpu().numpy()[0, :, :]
        segm = cv2.resize(small_seg.astype(np.uint8), (original_w, original_h))
        
        return segm
