from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

auth_provider = PlainTextAuthProvider("cassandra", "cassandra")
cluster = Cluster(["127.0.0.1"], port=9042, auth_provider=auth_provider)
session = cluster.connect()
print("Connected to Cassandra!")
