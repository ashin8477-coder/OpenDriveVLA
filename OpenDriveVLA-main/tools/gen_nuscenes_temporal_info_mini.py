import argparse
import json
import os
import pickle


def filter_infos(root, version, in_path, out_path):
    scene_json = os.path.join(root, version, 'scene.json')
    if not os.path.exists(scene_json):
        raise FileNotFoundError(f'Scene metadata not found: {scene_json}')

    with open(scene_json, 'r') as f:
        scenes = json.load(f)
    scene_tokens = {scene['token'] for scene in scenes}

    with open(in_path, 'rb') as f:
        data = pickle.load(f)

    infos = [info for info in data['infos'] if info.get('scene_token') in scene_tokens]
    metadata = data.get('metadata', {})
    metadata['version'] = version

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'infos': infos, 'metadata': metadata}, f)
    print(f'Saved {len(infos)} infos to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Filter temporal nuScenes infos for mini split')
    parser.add_argument('--root', required=True, help='Path to nuScenes root (contains v1.0-mini etc.)')
    parser.add_argument('--version', default='v1.0-mini', help='nuScenes version, e.g., v1.0-mini')
    parser.add_argument('--in-train', default='/workspace/OpenDriveVLA/data/infos/nuscenes_infos_temporal_train.pkl')
    parser.add_argument('--in-val', default='/workspace/OpenDriveVLA/data/infos/nuscenes_infos_temporal_val.pkl')
    parser.add_argument('--out-dir', default='/workspace/OpenDriveVLA/data/infos_mini', help='Output directory')
    args = parser.parse_args()

    out_train = os.path.join(args.out_dir, 'nuscenes_infos_temporal_train.pkl')
    out_val = os.path.join(args.out_dir, 'nuscenes_infos_temporal_val.pkl')

    filter_infos(args.root, args.version, args.in_train, out_train)
    filter_infos(args.root, args.version, args.in_val, out_val)


if __name__ == '__main__':
    main()
