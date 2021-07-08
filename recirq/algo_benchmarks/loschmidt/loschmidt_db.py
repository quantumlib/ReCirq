from sqlalchemy import create_engine

from recirq.otoc.loschmidt import TiltedSquareLatticeLoschmidtSpec

engine = create_engine("sqlite+pysqlite:///:memory:", echo=True, future=True)

from sqlalchemy import MetaData

metadata = MetaData()

from sqlalchemy.orm import registry, Session

mapper_registry = registry()

from sqlalchemy import Column, Integer

from sqlalchemy.orm import declarative_base

RelationalBase = declarative_base()


class TiltedSquareLatticeLoschmidtSpecRelational(RelationalBase):
    __tablename__ = 'diagonal_rectangle_loschmidt_spec'

    topology_width = Column(Integer, nullable=False)
    topology_height = Column(Integer, nullable=False)
    macrocycle_depth = Column(Integer, nullable=False)
    instance_i = Column(Integer, nullable=False)
    n_repetitions = Column(Integer, nullable=False)

    @classmethod
    def from_dataclass(cls, obj: TiltedSquareLatticeLoschmidtSpec):
        return cls(
            topology_width=obj.topology.width,
            topology_height=obj.topology.height,
            macrocycle_depth=obj.macrocycle_depth,
            instance_i=obj.instance_i,
            n_repetitions=obj.n_repetitions,
        )


RelationalBase.metadata.create_all()

with Session(engine) as session:
    session.commit()
