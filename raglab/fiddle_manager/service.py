import sqlite3
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class FiddleManager :
    def __init__(self, user_id) :
        self.user_id

    def create(name) :
        engine = create_engine('sqlite:///userdb.db', echo=True)
        
        new_user = FiddleView(username=self.user_id, email='john@example.com')
        session.add(new_user)
        session.commit()

    def get(id) :
        pass

    def update(id, value) :
        pass

    def delete(id) :
        pass


Base = declarative_base()

class FiddleView(Base):
    __tablename__ = 'fiddle_views'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    json = Column(String)


# Créer les tables dans la base de données
Base.metadata.create_all(engine)

# Créer une session
Session = sessionmaker(bind=engine)
session = Session()

