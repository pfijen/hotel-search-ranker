mkdir -p ~/.streamlit/

echo "
[general]
email = "paulo.fijen@gmail.com"
" > ~/.streamlit/credentials.toml

echo "
[server]
headless = true
enableCORS = false
port = $PORT
" > ~/.streamlit/config.toml
