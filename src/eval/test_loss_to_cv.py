import click
from os.path import join as j
from tqdm import tqdm
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@click.command()
@click.argument('base_folder')
@click.argument('output_file')
def main(base_folder,output_file):

    #Do not load images or other data to save time and memory
    tf_size_guidance = {
        'compressedHistograms': 0,
        'images': 0,
        'histograms': 0
    }

    search_path = j(base_folder,'*.ecompute*')
    file_paths = glob(search_path)

    with open(output_file,'w') as fout:
        for file_path in tqdm(file_paths):
            print(file_path)
            event_acc = EventAccumulator(file_path, tf_size_guidance)
            event_acc.Reload()

            test_loss = event_acc.Scalars('test/test_loss')
            train_loss = event_acc.Scalars('test/train_loss')

            steps = len(test_loss)

            fout.write(file_path)
            fout.write(',%s'%('test'),)
            for i in range(steps):
                fout.write(",%s"%(test_loss[i][2],))
            fout.write("\n")

            fout.write(file_path)
            fout.write(',%s'%('train'))
            for i in range(steps):
                fout.write(",%s"%(train_loss[i][2],))
            fout.write("\n")

if __name__ == '__main__':
    main()
