import cv2
import numpy as np
import os
import tempfile
import urllib.request


# 使用临时英文目录，而不是默认的 keras 缓存目录
def get_file_safe(fname, url):
    """安全下载文件到英文路径"""
    # 使用 Windows 临时目录（通常是 C:\Users\用户名\AppData\Local\Temp）
    # 这个路径不包含中文
    cache_dir = os.path.join(tempfile.gettempdir(), 'paz_models')
    os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(cache_dir, fname)

    if not os.path.exists(local_path):
        print(f"下载模型到: {local_path}")
        urllib.request.urlretrieve(url, local_path)
        print("下载完成")

    return local_path


class HaarCascadeDetector(object):
    """Haar cascade face detector.

    # Arguments
        weights: String. Postfix to default openCV haarcascades XML files, see [1]
            e.g. `eye`, `frontalface_alt2`, `fullbody`
        class_arg: Int. Class label argument.
        scale = Float. Scale for image reduction
        neighbors: Int. Minimum neighbors

    # Reference
        - [Haar
            Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
    """

    def __init__(self, weights='frontalface_default', class_arg=None,
                 scale=1.3, neighbors=5):
        self.weights = weights
        self.name = 'haarcascade_' + weights + '.xml'
        self.url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{self.name}'

        # 使用安全路径（英文临时目录）
        self.path = get_file_safe(self.name, self.url)

        # 加载模型
        self.model = cv2.CascadeClassifier(self.path)
        if self.model.empty():
            raise RuntimeError(f"无法加载 Haar Cascade 模型: {self.path}")

        self.class_arg = class_arg
        self.scale = scale
        self.neighbors = neighbors
        print(f"✅ 成功加载模型: {self.path}")

    def __call__(self, gray_image):
        """ Detects faces from gray images.

        # Arguments
            gray_image: Numpy array of shape ``(H, W, 2)``.

        # Returns
            Numpy array of shape ``(num_boxes, 4)``.
        """
        if len(gray_image.shape) != 2:
            raise ValueError('Invalid gray image shape:', gray_image.shape)
        args = (gray_image, self.scale, self.neighbors)
        boxes = self.model.detectMultiScale(*args)
        boxes_point_form = np.zeros_like(boxes)
        if len(boxes) != 0:
            boxes_point_form[:, 0] = boxes[:, 0]
            boxes_point_form[:, 1] = boxes[:, 1]
            boxes_point_form[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes_point_form[:, 3] = boxes[:, 1] + boxes[:, 3]
            if self.class_arg is not None:
                class_args = np.ones((len(boxes_point_form), 1))
                class_args = class_args * self.class_arg
                boxes_point_form = np.hstack((boxes_point_form, class_args))
        return boxes_point_form.astype('int')