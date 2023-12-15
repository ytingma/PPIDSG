import torch
import random
import numpy as np
_rotate = []
_negaposi = []
_reverse = []
_channel = []
random.seed(30)
_shf = []


def rotate(val, p):
  out = val
  if p == 0:
    for i in range(3):
      out[:, :, :, i] = np.rot90(val[:, :, :, i], k=1, axes=(1, 2))
  elif p == 1:
    for i in range(3):
      out[:, :, :, i] = np.rot90(val[:, :, :, i], k=2, axes=(1, 2))
  elif p == 2:
    for i in range(3):
      out[:, :, :, i] = np.rot90(val[:, :, :, i], k=3, axes=(1, 2))
  return out


def negaposi(val, o):
  out = val
  p = np.full((val.shape[0], 4, 4, 3), 255) # 4: block size; 3: channel number
  if o == 0:
    for i in range(3):
      out[:, :, :, i] = p[:, :, :, i] - val[:, :, :, i]
  return out


def reverse(val, p):
  out = val
  if p == 0:
    for i in range(3):
      out[:, :, :, i] = np.flip(val[:, :, :, i], axis=1)
  elif p == 1:
    for i in range(3):
      out[:, :, :, i] = np.flip(val[:, :, :, i], axis=2)
  return out


def channel_change(val, p):
  out = val
  if p == 1:
    out[:, :, :, 0] = val[:, :, :, 1]
    out[:, :, :, 1] = val[:, :, :, 0]
  elif p == 2:
    out[:, :, :, 0] = val[:, :, :, 2]
    out[:, :, :, 2] = val[:, :, :, 0]
  elif p == 3:
    out[:, :, :, 1] = val[:, :, :, 2]
    out[:, :, :, 2] = val[:, :, :, 1]
  elif p == 4:
    out[:, :, :, 0] = val[:, :, :, 2]
    out[:, :, :, 1] = val[:, :, :, 0]
    out[:, :, :, 2] = val[:, :, :, 1]
  elif p == 5:
    out[:, :, :, 0] = val[:, :, :, 1]
    out[:, :, :, 1] = val[:, :, :, 2]
    out[:, :, :, 2] = val[:, :, :, 0]
  return out


# 64: the block number in IS(need to change when the block size change)
for i in range(64):
  x = random.randint(0, 3)
  z = random.randint(0, 2)
  a = random.randint(0, 5)
  if i % 2 == 0:
      _negaposi.append(1)
  else:
      _negaposi.append(0)
  _rotate.append(x)
  _reverse.append(z)
  _channel.append(a)
  _shf.append(i)
random.shuffle(_shf)
random.shuffle(_negaposi)


# block size: 2
def EtC_cifar2(img):
    img = np.transpose(img * 255.0, (0, 2, 3, 1))

    for i in range(16):
      for j in range(16):
        img[:, i*2:(i+1)*2, j*2:(j+1)*2, :] = rotate(img[:, i*2:(i+1)*2, j*2:(j+1)*2, :], _rotate[i*16+j])
        img[:, i*2:(i+1)*2, j*2:(j+1)*2, :] = negaposi(img[:, i*2:(i+1)*2, j*2:(j+1)*2, :], _negaposi[i*16+j])
        img[:, i*2:(i+1)*2, j*2:(j+1)*2, :] = reverse(img[:, i*2:(i+1)*2, j*2:(j+1)*2, :], _reverse[i*16+j])
        img[:, i*2:(i+1)*2, j*2:(j+1)*2, :] = channel_change(img[:, i*2:(i+1)*2, j*2:(j+1)*2, :], _channel[i*16+j])
    tmp = img.copy()

    for i in range(256):
      l = _shf[i]//16
      r = _shf[i] % 16
      a = i//16
      b = i % 16
      img[:, a*2:(a+1)*2, b*2:(b+1)*2, :] = tmp[:, l*2:(l+1)*2, r*2:(r+1)*2, :].copy()
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))/255.0)
    return img


# block size: 8
def EtC_cifar8(img):
    img = np.transpose(img * 255.0, (0, 2, 3, 1))

    for i in range(4):
      for j in range(4):
        img[:, i*8:(i+1)*8, j*8:(j+1)*8, :] = rotate(img[:, i*8:(i+1)*8, j*8:(j+1)*8, :], _rotate[i*4+j])
        img[:, i*8:(i+1)*8, j*8:(j+1)*8, :] = negaposi(img[:, i*8:(i+1)*8, j*8:(j+1)*8, :], _negaposi[i*4+j])
        img[:, i*8:(i+1)*8, j*8:(j+1)*8, :] = reverse(img[:, i*8:(i+1)*8, j*8:(j+1)*8, :], _reverse[i*4+j])
        img[:, i*8:(i+1)*8, j*8:(j+1)*8, :] = channel_change(img[:, i*8:(i+1)*8, j*8:(j+1)*8, :], _channel[i*4+j])
    tmp = img.copy()

    for i in range(16):
      l = _shf[i]//4
      r = _shf[i] % 4
      a = i//4
      b = i % 4
      img[:, a*8:(a+1)*8, b*8:(b+1)*8, :] = tmp[:, l*8:(l+1)*8, r*8:(r+1)*8, :].copy()
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))/255.0)
    return img


# block size: 4
def EtC_cifar4(img):
    img = np.transpose(img * 255.0, (0, 2, 3, 1))

    for i in range(8):
        for j in range(8):
          img[:, i*4:(i+1)*4, j*4:(j+1)*4, :] = rotate(img[:, i*4:(i+1)*4, j*4:(j+1)*4, :], _rotate[i*8+j])
          img[:, i*4:(i+1)*4, j*4:(j+1)*4, :] = negaposi(img[:, i*4:(i+1)*4, j*4:(j+1)*4, :], _negaposi[i*8+j])
          img[:, i*4:(i+1)*4, j*4:(j+1)*4, :] = reverse(img[:, i*4:(i+1)*4, j*4:(j+1)*4, :], _reverse[i*8+j])
          img[:, i*4:(i+1)*4, j*4:(j+1)*4, :] = channel_change(img[:, i*4:(i+1)*4, j*4:(j+1)*4, :], _channel[i*8+j])
    tmp = img.copy()

    for i in range(64):
      l = _shf[i]//8
      r = _shf[i] % 8
      a = i//8
      b = i % 8
      img[:, a*4:(a+1)*4, b*4:(b+1)*4, :] = tmp[:, l*4:(l+1)*4, r*4:(r+1)*4, :].copy()
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))/255.0)
    return img


# block size: 16
def EtC_cifar16(img):
  img = np.transpose(img * 255.0, (0, 2, 3, 1))

  for i in range(2):
    for j in range(2):
      img[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = rotate(img[:, i*16:(i+1)*16, j*16:(j+1)*16, :], _rotate[i*2+j])
      img[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = negaposi(img[:, i*16:(i+1)*16, j*16:(j+1)*16, :], _negaposi[i*2+j])
      img[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = reverse(img[:, i*16:(i+1)*16, j*16:(j+1)*16, :], _reverse[i*2+j])
      img[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = channel_change(img[:, i*16:(i+1)*16, j*16:(j+1)*16, :], _channel[i*2+j])
    tmp = img.copy()

  for i in range(4):
    l = _shf[i] // 2
    r = _shf[i] % 2
    a = i // 2
    b = i % 2
    img[:, a * 16:(a + 1) * 16, b * 16:(b + 1) * 16, :] = tmp[:, l * 16:(l + 1) * 16, r * 16:(r + 1) * 16, :].copy()
  img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)) / 255.0)
  return img
