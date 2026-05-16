"""
Backend MFB — Bureau d'Ordre Digital
API Flask qui reçoit un PDF, appelle l'agent IA, et retourne les 13 champs extraits.

Usage:
    pip install flask flask-cors anthropic
    set ANTHROPIC_API_KEY=sk-ant-...
    python server.py
"""

import os
import base64
import json
import anthropic
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── App Flask ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Autorise les appels depuis le portail Netlify (cross-origin)

# ─── Client Anthropic ────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # Lit ANTHROPIC_API_KEY depuis l'environnement

# ─── Prompt système (mis en cache) ───────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un agent spécialisé dans l'extraction de données de factures de transport pour Maroc Fruit Board (MFB).

Tu reçois une facture en PDF et tu dois extraire exactement les 13 champs suivants. Retourne UNIQUEMENT un objet JSON valide, sans texte autour, sans markdown, sans commentaires.

Champs à extraire :
1.  "num_facture"           — Numéro de la facture (ex: "INV-2024-001")
2.  "fournisseur"           — Nom du fournisseur / transporteur émetteur
3.  "num_commande"          — Numéro de commande ou bon de commande MFB (peut être null)
4.  "mode_transport"        — Mode : "Maritime", "Aérien" ou "Routier"
5.  "navire_vehicule"       — Nom du navire, véhicule ou numéro de vol (peut être null)
6.  "pol"                   — Port/Aéroport/Ville de chargement (Point of Loading)
7.  "pod"                   — Port/Aéroport/Ville de déchargement (Point of Discharge)
8.  "num_bl_cmr_lta"        — Numéro BL (maritime), CMR (routier) ou LTA (aérien) (peut être null)
9.  "nbre_unites"           — Nombre d'unités de transport (conteneurs, camions, palettes…)
10. "lieu_enlevement"       — Lieu d'enlèvement / prise en charge des marchandises
11. "date_depart"           — Date de départ au format ISO 8601 "YYYY-MM-DD" (peut être null)
12. "date_comptabilisation" — Date de comptabilisation au format ISO 8601 "YYYY-MM-DD"
13. "montant"               — Montant total de la facture en nombre décimal (ex: 15000.00)

Règles :
- Si un champ est introuvable, mets null (pas de chaîne vide).
- Pour le montant, retourne uniquement le nombre, sans devise ni symbole.
- Pour les dates, convertis tout format courant en YYYY-MM-DD.
- Pour mode_transport, déduis-le du contexte si non mentionné (BL → Maritime, LTA → Aérien, CMR → Routier).
"""

# ─── Extraction via Claude ────────────────────────────────────────────────────
def extraire_depuis_bytes(pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extrais les 13 champs MFB de cette facture et retourne uniquement le JSON.",
                    },
                ],
            }
        ],
    )

    # Récupère le bloc texte (ignore les blocs thinking)
    texte = ""
    for block in response.content:
        if block.type == "text":
            texte = block.text.strip()
            break

    # Nettoie les backticks éventuels
    if texte.startswith("```"):
        lignes = texte.split("\n")
        texte = "\n".join(lignes[1:-1]).strip()

    return json.loads(texte)


# ─── Endpoint : import d'une facture PDF ─────────────────────────────────────
@app.route("/api/import-facture", methods=["POST"])
def import_facture():
    """
    Reçoit un fichier PDF (multipart/form-data, champ "pdf"),
    extrait les 13 champs MFB via Claude, et retourne le JSON.
    """
    if "pdf" not in request.files:
        return jsonify({"erreur": "Aucun fichier PDF reçu (champ attendu : 'pdf')"}), 400

    fichier = request.files["pdf"]

    if fichier.filename == "":
        return jsonify({"erreur": "Nom de fichier vide"}), 400

    if not fichier.filename.lower().endswith(".pdf"):
        return jsonify({"erreur": "Le fichier doit être un PDF"}), 400

    pdf_bytes = fichier.read()
    if len(pdf_bytes) == 0:
        return jsonify({"erreur": "Fichier PDF vide"}), 400

    try:
        data = extraire_depuis_bytes(pdf_bytes)

        # Champs requis — null si absents
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

    except json.JSONDecodeError as e:
        return jsonify({"erreur": f"Réponse Claude non parseable : {str(e)}"}), 500
    except anthropic.APIError as e:
        return jsonify({"erreur": f"Erreur API Claude : {str(e)}"}), 502
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


# ─── Endpoint : santé du serveur ─────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """Vérifie que le serveur tourne et que la clé API est configurée."""
    api_key_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return jsonify({
        "statut": "ok",
        "api_key_configuree": api_key_ok,
        "modele": "claude-opus-4-7",
    }), 200


# ─── Démarrage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[MFB Backend] Démarrage sur http://localhost:{port}")
    print(f"[MFB Backend] Clé API : {'✓ configurée' if os.environ.get('ANTHROPIC_API_KEY') else '✗ MANQUANTE — définis ANTHROPIC_API_KEY'}")
    app.run(host="0.0.0.0", port=port, debug=False)
