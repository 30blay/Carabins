from sqlalchemy import MetaData, create_engine
from SQLModels import metadata, Subject


meta = MetaData()

dbName = 'foo.db'

engine = create_engine('sqlite:///' + dbName)
metadata.create_all(engine)
