'''
    Extracts vein length from manual measurements in csv file
    provided by Ashley

    See notebooks/ComareToData.ipynb for exploration of this process.
'''
import dataset
import click
from tqdm import tqdm
from csv import DictReader

@click.command()
@click.argument('database_conn')
@click.argument('table_name')
@click.argument('csv_file')
def main(csv_file,database_conn,table_name):
    db = dataset.connect(database_conn)

    table = db[table_name]
    table.delete()
    with open(csv_file) as fin:
        reader = DictReader(fin)
        for row in tqdm(reader):
            if row['Vein Length (um)'] != '':
                filename = row['File Name'].split('-')
                plant = filename[0]
                replicate = filename[1]
                length = row['Vein Length (um)']
                table.insert(dict(plant=plant, replicate=replicate, length=float(length)))
                #print("Plant: %s, Replicate: %s, Length: %s"%(plant, replicate, length))

if __name__ == '__main__':
    main()
