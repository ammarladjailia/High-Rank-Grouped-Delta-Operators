"""
Patch script to update French axis labels and title to English in beta_distributions.png.
Translates:
- 'Valeur de Beta' -> 'Gate Value (β)'
- 'Densité' -> 'Density'
- 'Distribution effective des portes (Beta) en Inférence' -> 'Effective Gate (β) Distribution at Inference'
"""

import os
from PIL import Image, ImageDraw, ImageFont

def patch_beta_distributions_image(src_img_path, target_paths=None):
    if not os.path.exists(src_img_path):
        raise FileNotFoundError(f"Source image not found: {src_img_path}")

    im = Image.open(src_img_path).convert('RGBA')

    # Fonts
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font_label = ImageFont.truetype(font_path, 42)
    font_title = ImageFont.truetype(font_path, 58)

    draw = ImageDraw.Draw(im)

    # 1. Clear Title area (French: "Distribution effective des portes (Beta) en Inférence")
    draw.rectangle([(850, 140), (2220, 210)], fill=(255, 255, 255, 255))
    title_text = 'Effective Gate (β) Distribution at Inference'
    tb = font_title.getbbox(title_text)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    tx = int(1537.5 - tw / 2)
    ty = int(175 - th / 2) - tb[1]
    draw.text((tx, ty), title_text, fill=(0, 0, 0, 255), font=font_title)

    # 2. Clear X-label area (French: "Valeur de Beta")
    draw.rectangle([(1300, 1680), (1750, 1740)], fill=(255, 255, 255, 255))
    xlabel_text = 'Gate Value (β)'
    xb = font_label.getbbox(xlabel_text)
    xw = xb[2] - xb[0]
    xh = xb[3] - xb[1]
    xx = int(1537.5 - xw / 2)
    xy = int(1706 - xh / 2) - xb[1]
    draw.text((xx, xy), xlabel_text, fill=(0, 0, 0, 255), font=font_label)

    # 3. Clear Y-label area (French: "Densité")
    draw.rectangle([(185, 820), (240, 1010)], fill=(255, 255, 255, 255))
    ylabel_text = 'Density'
    yb = font_label.getbbox(ylabel_text)
    yw = yb[2] - yb[0]
    yh = yb[3] - yb[1]

    # Create a transparent surface for the rotated text
    txt_img = Image.new('RGBA', (yw + 20, yh + 20), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((10 - yb[0], 10 - yb[1]), ylabel_text, fill=(0, 0, 0, 255), font=font_label)
    rotated_txt = txt_img.rotate(90, expand=True)

    rx = int(212.5 - rotated_txt.width / 2)
    ry = int(909 - rotated_txt.height / 2)
    im.paste(rotated_txt, (rx, ry), rotated_txt)

    if target_paths is None:
        target_paths = [src_img_path]

    for p in target_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        im.save(p)
        print(f"Updated: {p}")

if __name__ == '__main__':
    targets = [
        'Revision/Tex_Revision/figures/beta_distributions.png',
        'outputs/beta_distributions.png',
        'Review_Color/figures/beta_distributions.png',
        'tmp_unzip/Tex_Revision/figures/beta_distributions.png',
    ]
    # Filter for existing directories
    valid_targets = [p for p in targets if os.path.exists(os.path.dirname(p))]
    patch_beta_distributions_image('Revision/Tex_Revision/figures/beta_distributions.png', valid_targets)
