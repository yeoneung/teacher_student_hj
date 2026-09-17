"""Render every final manuscript page; human visual review remains separate."""
import datetime
import json
from pathlib import Path
import fitz
from .study import digest


def main():
    pdf=Path('build/hj_teacher_transport_v7.pdf')
    root=Path('build/hj_transport_visual_review');root.mkdir(exist_ok=True)
    records=[]
    with fitz.open(pdf) as document:
        for i,page in enumerate(document):
            path=root/f'page_{i+1:03d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(path)
            records.append(dict(page=i+1,path=str(path),sha256=digest(path),
                width_points=page.rect.width,height_points=page.rect.height))
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pdf_sha256=digest(pdf),pages=len(records),renders=records,
        scope='Rendering only. This manifest is not a claim that any page has been visually reviewed.')
    (root/'render_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(pages=len(records),root=str(root),pdf_sha256=result['pdf_sha256']),indent=2))


if __name__=='__main__':main()
