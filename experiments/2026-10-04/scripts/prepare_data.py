import hashlib,json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
STORE=Path('/EXTERNAL_RESOURCE/9edba807c586')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    STORE.mkdir(exist_ok=False)
    (STORE/'checkpoints').mkdir();(STORE/'data').mkdir()
    (ROOT/'data').symlink_to(STORE/'data',target_is_directory=True)
    # Replace only our newly created, empty checkpoint directory with a link.
    (ROOT/'checkpoints').rmdir();(ROOT/'checkpoints').symlink_to(STORE/'checkpoints',target_is_directory=True)
    packed=Path('/EXTERNAL_RESOURCE/9755c793eca0')
    manifest=json.loads((packed/'manifest.json').read_text())
    rng=np.random.default_rng(20261004)
    chosen=np.linspace(0,len(manifest['shards'])-1,16,dtype=int)
    train=[];provenance=[]
    for shard_index in chosen:
        s=manifest['shards'][shard_index];p=packed/s['path']
        digest=sha(p)
        assert digest==s['sha256'],p
        blocks=np.memmap(p,dtype='<u2',mode='r').reshape(-1,1024)
        indices=rng.choice(len(blocks),300,replace=False)
        for block_index in indices:
            start=int(rng.integers(0,768));values=blocks[block_index,start:start+257].copy()
            train.append(values)
            provenance.append({'kind':'original_v7_training_packed','shard':str(p),
                               'shard_sha256':digest,'block':int(block_index),'start':start,
                               'source_manifest':str(packed/'manifest.json'),
                               'license':'original corpus mixture; see source dataset manifests; no redistribution'})
    order=rng.permutation(4800)
    train=np.stack(train)[order]
    provenance=[provenance[int(i)] for i in order]
    tokenizer_path=Path('/EXTERNAL_RESOURCE/f033a331c535')
    tokenizer=Tokenizer.from_file(str(tokenizer_path)); assert tokenizer.get_vocab_size()==32768
    val_path=tokenizer_path.parents[1]/'shards/val.jsonl'
    selected=set(np.random.default_rng(999).choice(87335,256,replace=False).tolist())
    val=[];val_provenance=[]
    with val_path.open() as f:
        for i,line in enumerate(f):
            if i not in selected:continue
            row=json.loads(line);tokens=tokenizer.encode(row['text']).ids[:1022]+[2]
            if len(tokens)<257:continue
            val.append(np.array(tokens[:257],dtype=np.uint16))
            val_provenance.append({'line':i+1,'dataset':row.get('_boothgpt_dataset_id'),
                                   'source':row.get('_boothgpt_source_path'),
                                   'version':row.get('_boothgpt_version'),
                                   'license':'preserved original dataset; see original manifest'})
            if len(val)==128:break
    assert len(val)==128
    val=np.stack(val)
    train_hashes={hashlib.sha256(row.tobytes()).hexdigest() for row in train}
    assert not any(hashlib.sha256(row.tobytes()).hexdigest() in train_hashes for row in val)
    for name,values in [('train',train),('validation',val)]:
        assert values.max()<32768
        np.save(STORE/'data'/f'{name}.npy',values)
    (STORE/'data'/'train_provenance.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in provenance))
    (STORE/'data'/'validation_provenance.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in val_provenance))
    meta={'train_shape':list(train.shape),'val_shape':list(val.shape),
          'train_loss_tokens':4800*256,'val_loss_tokens':128*256,
          'seed':20261004,'training_shard_indices':chosen.tolist(),
          'dataset':'original v7 16 evenly spaced packed shards; 300 distinct blocks/shard',
          'validation':'first 128 eligible of 256 uniformly sampled original validation documents',
          'validation_source_counts':{key:sum(p['dataset']==key for p in val_provenance)
                                      for key in set(p['dataset'] for p in val_provenance)},
          'train_sha256':sha(STORE/'data'/'train.npy'),
          'validation_sha256':sha(STORE/'data'/'validation.npy'),
          'tokenizer_path':str(tokenizer_path),'tokenizer_sha256':sha(tokenizer_path),
          'packed_manifest_path':str(packed/'manifest.json'),
          'packed_manifest_sha256':sha(packed/'manifest.json'),
          'split_manifest_sha256':sha(val_path.parent/'manifest.json'),
          'exact_train_validation_window_overlap':0,
          'limitations':'original train/val split; no new semantic deduplication; old v7 rather than current Fedora v2 mix'}
    (ROOT/'configs'/'data.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
