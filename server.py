import os
import base64
import json
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

SYSTEM_PROMPT = """Tu es un agent spécialisé dans l'extraction de données de factures de transport pour Maroc Fruit Board (MFB).
Extrais exactement ces 13 champs du PDF. Retourne UNIQUEMENT un objet JSON valide, sans texte autour, sans markdown.
Champs à extraire :
1.  "num_facture"           — Numéro de la facture
2.  "fournisseur"           — Nom du fournisseur / transporteur
3.  "num_commande"          — Numéro de commande MFB (null si absent)
4.  "mode_transport"        — "Maritime", "Aérien" ou "Routier"
5.  "navire_vehicule"       — Nom du navire, véhicule ou vol (null si absent)
6.  "pol"                   — Port/Ville de chargement
7.  "pod"                   — Port/Ville de déchargement
8.  "num_bl_cmr_lta"        — Numéro BL / CMR / LTA (null si absent)
9.  "nbre_unites"           — Nombre d'unités de transport
10.⁠ ⁠"lieu_enlevement"       — Lieu d'enlèvement des marchandises
11.⁠ ⁠"date_depart"           — Date départ format YYYY-MM-DD (null si absent)
12.⁠ ⁠"date_comptabilisation" — Date comptabilisation format YYYY-MM-DD
13.⁠ ⁠"montant"               — Montant total en nombre décimal (ex: 15000.00)
Règles : champ introuvable = null, montant sans devise, dates en YYYY-MM-DD."""


def extraire_depuis_bytes(pdf_bytes: bytes) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "Extrais les 13 champs MFB de cette facture et retourne uniquement le JSON.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    texte = response.text.strip()
    if texte.startswith("```"):
        lignes = texte.split("\n")
        texte = "\n".join(lignes[1:-1]).strip()

    return json.loads(texte)


@app.route("/api/import-facture", methods=["POST"])
def import_facture():
    if "pdf" not in request.files:
        return jsonify({"erreur": "Aucun fichier PDF reçu"}), 400

    fichier = request.files["pdf"]
    if not fichier.filename.lower().endswith(".pdf"):
        return jsonify({"erreur": "Le fichier doit être un PDF"}), 400

    pdf_bytes = fichier.read()
    if len(pdf_bytes) == 0:
        return jsonify({"erreur": "Fichier PDF vide"}), 400

    try:
        data = extraire_depuis_bytes(pdf_bytes)
        champs = [
            "num_facture", "fournisseur", "num_commande", "mode_transport",
            "navire_vehicule", "pol", "pod", "num_bl_cmr_lta", "nbre_unites",
            "lieu_enlevement", "date_depart", "date_comptabilisation", "montant",
        ]
        for c in champs:
            if c not in data:
                data[c] = None
        data["_fichier"] = fichier.filename
        data["_statut"] = "ok"
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    api_key_ok = bool(os.environ.get("GOOGLE_API_KEY"))
    return jsonify({
        "statut": "ok",
        "updated": "yes",
        "api_key_configuree": api_key_ok,
        "modele": "gemini-2.5-flash",
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)