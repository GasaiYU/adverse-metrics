# How to run SSIM metric

## 1. Prepare Filelist

You should input the real and synthetic filelist. The filelist's format is as below:

```
image_path(.jpg)
```

## 2. Run PSNR

Run as below in the same dir of the `psnr.py`

```bash
python3 psnr.py --real_filelist <fake filelist> --real_filelist <real filelist> [--subset_size <subset_size>]
```

The result will be printed in the terminal.