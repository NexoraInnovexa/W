"""Add Pitch model

Revision ID: b73052736f9f
Revises: 844c635b6042
Create Date: 2024-12-21 14:52:34.564324

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b73052736f9f'
down_revision = '844c635b6042'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('pitches',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('startup_id', sa.Integer(), nullable=False),
    sa.Column('investor_id', sa.Integer(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['investor_id'], ['investor.id'], ),
    sa.ForeignKeyConstraint(['startup_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('pitches')
    # ### end Alembic commands ###
