import gdown

checkpoint_dir = './pretrained_weights/'
model_restore = checkpoint_dir + 'inference.pth'
url = 'https://drive.google.com/u/0/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9'
gdown.download(url, model_restore, quiet=False)
