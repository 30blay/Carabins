from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Date, Integer, String, Float, Boolean, PrimaryKeyConstraint
from sqlalchemy.orm import relationship


Base = declarative_base()
metadata = Base.metadata


class Subject(Base):
    __tablename__ = 'subject'

    subject_id = Column(Integer, primary_key=True)


# Cette table regroupe les réponse des sujets à des formulaires standards MFI (Multidimentional Fatigue Inventory)
# pain: douleur sur une échelle de 1 à 10
# gen : fatigue générale
# phys : fatigue physique
# men : fatigue mentale
# mot : motivation réduite
# act : activités réduites
class Fatigue(Base):
    __tablename__ = 'fatigue'

    subject_id = Column(Integer, ForeignKey('subject.subject_id'), primary_key=True)
    date = Column(Date, nullable=True)

    injury = Column(String(250), nullable=True)
    pain = Column(Integer)
    gen = Column(Integer)
    phys = Column(Integer)
    men = Column(Integer)
    act = Column(Integer)
    mot = Column(Integer)


class Medical(Base):
    __tablename__ = 'medical'

    subject_id = Column(Integer, ForeignKey('subject.subject_id'), primary_key=True)
    # TODO make nullable false
    position = Column(String(5), nullable=True)
    status = Column(String(1), nullable=True)
    height = Column(Float)
    weight = Column(Float)
    fat = Column(Float)
    arm_length = Column(Float)
    asym_drop_box = Column(Boolean)
    asym_tuck_jump = Column(Boolean)
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


class Handwriting(Base):
    __tablename__ = 'handwriting'

    subject_id = Column(Integer, ForeignKey('subject.subject_id'))
    test_id = Column(Integer)
    PrimaryKeyConstraint(subject_id, test_id)
    t0 = Column(Float)
    D1 = Column(Float)
    mu1 = Column(Float)
    ss1 = Column(Float)
    D2 = Column(Float)
    mu2 = Column(Float)
    ss2 = Column(Float)
    SNR = Column(Float)
