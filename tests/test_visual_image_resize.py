import unittest

from PIL import Image

from downstream_task.report_generation_and_vqa.image_preprocess import resize_visual_image


class VisualImageResizeTests(unittest.TestCase):
    def test_uses_512_square_resize_for_len_vis_input_256(self):
        image = Image.new("RGB", (2190, 2048))

        resized = resize_visual_image(image, len_vis_input=256)

        self.assertEqual(resized.size, (512, 512))

    def test_uses_224_square_resize_for_small_visual_input(self):
        image = Image.new("RGB", (640, 480))

        resized = resize_visual_image(image, len_vis_input=49)

        self.assertEqual(resized.size, (224, 224))


if __name__ == "__main__":
    unittest.main()
