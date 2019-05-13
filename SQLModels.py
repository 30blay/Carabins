from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, String, Date, Float, Boolean

Base = declarative_base()
metadata = Base.metadata

class Subject(Base):
    __tablename__ = 'subject'

    id = Column(Integer, primary_key=True)
    date = Column(Date)

    injury = Column(String(250), nullable=True)
    pain = Column(Integer)
    gen = Column(Integer)
    phys = Column(Integer)
    men = Column(Integer)
    act = Column(Integer)
    mot = Column(Integer)

    position = Column(String(5), nullable=False)
    status = Column(String(1), nullable=False)
    size = Column(Float)
    weight = Column(Float)
    fat = Column(Float)
    arm_length = Column(Float)
    asym_drop_box = Column(Boolean)
    asym_tuck_box = Column(Boolean)
    hop_g1 = Column(Float)
    hop_g2 = Column(Float)
    hop_d1 = Column(Float)
    hop_d2 = Column(Float)
    hop3_g1 = Column(Float)
    hop3_g2 = Column(Float)
    hop3_d1 = Column(Float)
    hop3_d2 = Column(Float)
    hop3_cr_g1 = Column(Float)
    hop3_cr_g2 = Column(Float)
    hop3_cr_d1 = Column(Float)
    hop3_cr_d2 = Column(Float)
    re_gh_g1 = Column(Float)
    re_gh_g2 = Column(Float)
    re_gh_d1 = Column(Float)
    re_gh_d2 = Column(Float)
    flex_g1 = Column(Float)
    flex_g2 = Column(Float)
    flex_d1 = Column(Float)
    flex_d2 = Column(Float)
    scap_g1 = Column(Float)
    scap_g2 = Column(Float)
    scap_d1 = Column(Float)
    scap_d2 = Column(Float)

