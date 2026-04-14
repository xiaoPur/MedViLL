def resolve_visual_image_size(len_vis_input):
    return 224 if len_vis_input < 100 else 512


def resize_visual_image(image, len_vis_input):
    target_size = resolve_visual_image_size(len_vis_input)
    return image.resize((target_size, target_size))
