#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    args = parser.parse_args()
    name = args.model_name.lower()
    if 'qwen3' in name:
        print('Qwen3DecoderLayer')
    elif 'qwen2' in name or 'qwen2.5' in name:
        print('Qwen2DecoderLayer')
    else:
        print('Qwen3DecoderLayer')

if __name__ == '__main__':
    main()
