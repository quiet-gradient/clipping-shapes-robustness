"""Prepare benchmark inputs from local official archives; no training or network calls."""
from pathlib import Path
import argparse,json

PACKAGE=Path(__file__).resolve().parents[1]

def link(source,target):
    source=source.resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.resolve()!=source:raise FileExistsError(f'{target} already exists; choose the existing archive explicitly')
    else:target.symlink_to(source)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtsrb-training-archive',type=Path)
    parser.add_argument('--gtsrb-test-images',type=Path)
    parser.add_argument('--gtsrb-test-labels',type=Path)
    args=parser.parse_args()
    if args.gtsrb_training_archive:
        from trustml.convnext_gtsrb.data import training_records,make_split
        source=args.gtsrb_training_archive.resolve()
        split=make_split(training_records(source),seed=42,target=4000,probe_seed=20260920)
        expected=json.loads((PACKAGE/'reports/ssg_gtsrb_v1/gtsrb_private_split_manifest.json').read_text())
        if split['split_hash']!=expected['split_hash'] or split['hashes']!=expected['hashes']:
            raise ValueError('regenerated train/dev/probe split differs from the frozen protocol')
        target=PACKAGE/'gtsrb_inputs';target.mkdir(exist_ok=True)
        link(source,target/'GTSRB_Final_Training_Images.zip')
        path=target/'splits.json'
        if path.exists() and json.loads(path.read_text())!=split:
            raise ValueError('existing split differs')
        path.write_text(json.dumps(split,sort_keys=True,indent=2)+'\n')
        print(json.dumps(dict(status='PASS',split_hash=split['split_hash'],counts={k:len(v) for k,v in split['indices'].items()})))
    for source,name in [(args.gtsrb_test_images,'GTSRB_Final_Test_Images.zip'),(args.gtsrb_test_labels,'GTSRB_Final_Test_GT.zip')]:
        if source:link(source,PACKAGE/'gtsrb_inputs'/name)
    if not any(vars(args).values()):parser.error('supply at least one official archive')

if __name__=='__main__':main()
