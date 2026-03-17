from http.server import SimpleHTTPRequestHandler, HTTPServer

class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <html>
            <head>
                <title>Pokédex</title>
                <style>
                    body {
                        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
                        font-family: 'Segoe UI', Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }
                    .container {
                        max-width: 900px;
                        margin: 40px auto;
                        background: #fff;
                        border-radius: 18px;
                        box-shadow: 0 6px 24px rgba(0,0,0,0.13);
                        padding: 32px 24px;
                        text-align: center;
                    }
                    h1 {
                        color: #ef5350;
                        margin-bottom: 20px;
                        font-size: 2.5em;
                        letter-spacing: 2px;
                    }
                    .poke-ball {
                        width: 60px;
                        margin-bottom: 18px;
                    }
                    .pokedex-table {
                        margin: 24px auto 0 auto;
                        border-collapse: collapse;
                        width: 100%;
                        font-size: 1.08em;
                    }
                    th, td {
                        border: 1px solid #ef5350;
                        padding: 8px 12px;
                        text-align: center;
                    }
                    th {
                        background: #ef5350;
                        color: #fff;
                    }
                    tr:nth-child(even) {
                        background: #f9f9f9;
                    }
                    .pokemon-img {
                        width: 70px;
                        filter: drop-shadow(0 4px 8px rgba(239,83,80,0.10));
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <img class="poke-ball" src="https://upload.wikimedia.org/wikipedia/commons/5/53/Poké_Ball_icon.svg" alt="Pokeball">
                    <h1>Pokédex</h1>
                    <table class="pokedex-table" id="pokedex">
                        <tr>
                            <th>Image</th>
                            <th>Name</th>
                            <th>Type(s)</th>
                        </tr>
                    </table>
                </div>
                <script>
                    async function fetchPokemon(id) {
                        const res = await fetch(`https://pokeapi.co/api/v2/pokemon/${id}/`);
                        return await res.json();
                    }
                    async function loadPokedex() {
                        const table = document.getElementById('pokedex');
                        for (let i = 1; i <= 20; i++) { // Show first 20 Pokémon
                            const data = await fetchPokemon(i);
                            const types = data.types.map(t => t.type.name).join(', ');
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td><img class="pokemon-img" src="${data.sprites.front_default}" alt="${data.name}"></td>
                                <td style="text-transform:capitalize">${data.name}</td>
                                <td>${types}</td>
                            `;
                            table.appendChild(row);
                        }
                    }
                    loadPokedex();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            super().do_GET()

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, MyHandler)
    print("Serving on http://localhost:8000")
    httpd.serve_forever()