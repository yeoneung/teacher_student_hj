"""Render a manuscript for actual visual review; do not certify it as reviewed."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default='build/hj_gridfree_rollout.pdf')
    parser.add_argument('--output', default='build/hj_cotangent_visual_review')
    args = parser.parse_args()
    source, target = Path(args.pdf), Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    page_paths, sheet_paths = [], []
    with fitz.open(source) as document:
        for index, page in enumerate(document):
            path = target / f'page_{index + 1:03d}.png'
            page.get_pixmap(dpi=120, alpha=False).save(path)
            page_paths.append(path)
        for offset in range(0, len(page_paths), 4):
            pages = []
            for index, path in enumerate(page_paths[offset:offset + 4], offset + 1):
                with Image.open(path) as original:
                    panel = original.convert('RGB')
                panel.thumbnail((744, 1053))
                labeled = Image.new('RGB', (764, 1087), '#d9dee3')
                labeled.paste(panel, ((764 - panel.width) // 2, 27))
                ImageDraw.Draw(labeled).text((10, 7), f'Page {index}', fill='black')
                pages.append(labeled)
            sheet = Image.new('RGB', (1528, 2174), '#d9dee3')
            for index, panel in enumerate(pages):
                sheet.paste(panel, ((index % 2) * 764, (index // 2) * 1087))
            path = target / f'sheet_{offset // 4 + 1:02d}.png'
            sheet.save(path)
            sheet_paths.append(str(path))
    record = dict(pdf=str(source), pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  pages=len(page_paths), page_images=[str(p) for p in page_paths],
                  contact_sheets=sheet_paths,
                  status='rendered; visual inspection remains required')
    (target / 'render_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
