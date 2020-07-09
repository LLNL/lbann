## 3D U-Net
LBANN implementation of the 3D U-Net:

> Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. "3D U-Net: learning dense volumetric segmentation from sparse annotation." In International conference on medical image computing and computer-assisted intervention, pp. 424-432, 2016.

This model requires Distconv.

### How to Train
```bash
python3 ./unet3d.py
```

See `python3 ./unet3d.py --help` for more options.
