"""
Backend MFB — Bureau d'Ordre Digital
"""
import os
import json
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'}
)

SYSTEM_PROMPT = """Tu es un agent spécialisé dans l'extraction de données de factures de transport pour Maroc Fruit Board (MFB).
Extrais exactement ces 13 champs du PDF. Retourne UNIQUEMENT un objet JSON valide, sans texte autour, sans markdown.
Champs : num_facture, fournisseur, num_commande, mode_transport, navire_vehicule, pol, pod, num_bl_cmr_lta, nbre_unites, lieu_enlevement, date_depart (YYYY-MM-DD), date_comptabilisation (YYYY-MM-DD), montant (nombre décimal).
Règles : champ introuvable = null, montant sans devise, dates en YYYY-MM-DD."""


def extraire_depuis_bytes(pdf_bytes: bytes) -> dict:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            SYSTEM_PROMPT,
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "Extrais les 13 champs MFB de cette facture et retourne uniquement le JSON.",
        ]
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
        champs = ["num_facture","fournisseur","num_commande","mode_transport",
                  "navire_vehicule","pol","pod","num_bl_cmr_lta","nbre_unites",
                  "lieu_enlevement","date_depart","date_comptabilisation","montant"]
        for c in champs:
            if c not in data:
                data[c] = None
        data["_fichier"] = fichier.filename
        data["_statut"] = "ok"
        return jsonify(data), 200
    except json.JSONDecodeError as e:
        return jsonify({"erreur": f"Réponse non parseable : {str(e)}"}), 500
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"statut": "ok", "modele": "gemini-1.5-flash"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
