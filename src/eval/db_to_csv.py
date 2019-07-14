'''
    Export database table data as:
        csv: Comma-separated values, first line contains column names.
        json: A JSON file containing a list of dictionaries for each row in the table. If a callback is given, JSON with padding (JSONP) will be generated.
        tabson:   Tabson is a smart combination of the space-efficiency of the CSV and the parsability and structure of JSON.
'''
import datafreeze
import dataset
import click

@click.command()
@click.argument('database_conn')
@click.argument('table_name')
@click.argument('format')
@click.argument('filename')
def main(database_conn,table_name, format, filename):
    db = dataset.connect(database_conn)
    result = db[table_name].all()
    datafreeze.freeze(result, format=format, filename=filename)

if __name__ == '__main__':
    main()
