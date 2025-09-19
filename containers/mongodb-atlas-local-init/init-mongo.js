// mongo-init/init-mongo.js
db = db.getSiblingDB('rag_dmr_db');         // Get or create the database 'mydatabase'
db.createCollection('rag_dmr_collection');  // Create the collection 'mycollection'
