# AI with memory

I made a trainer script (way before OpenAI did) which gives long-term memory for the neural network about the user.

The 2 validation files are the val_split.txt and the memory.txt (non-constant). memory.txt is a glorified log which enables the model to remember for eternity.

## Dependencies

- cuda
- pytorch
- os
- random
- mmap
- pickle

## Usage

Feel free to tune the script to fit your needs before running the training script.

Run:

`python trainer.py`
After running, make sure to input the mode (train or use)