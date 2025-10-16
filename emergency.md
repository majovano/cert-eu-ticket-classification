docker compose down
docker volume rm react-cert-eu-ml-challenge_postgres_data
mkdir temp
cd temp
git clone https://github.com/majovano/cert-eu-ticket-classification.git
cd cert-eu-ticket-classification
docker compose up -d