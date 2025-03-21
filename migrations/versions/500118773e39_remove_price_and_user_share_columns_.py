"""Remove price and user_share columns from Resource model

Revision ID: 500118773e39
Revises: 4b2e1dc9f242
Create Date: 2024-12-21 10:58:38.491916

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '500118773e39'
down_revision = '4b2e1dc9f242'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('resource', schema=None) as batch_op:
        batch_op.drop_column('price')
        batch_op.drop_column('user_share')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('resource', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_share', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False))
        batch_op.add_column(sa.Column('price', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False))

    # ### end Alembic commands ###
