"""
Agent IA MFB — Extraction automatique de factures PDF
Utilise l'API Claude (claude-opus-4-7) avec document blocks et prompt caching.

Usage:
    python extract_facture.py <chemin_pdf>
    python extract_facture.py facture.pdf

Retourne un JSON avec les 13 champs MFB extraits.
"""

import anthropic
import base64
import json
import sys
from pathlib import Path

# ─── Client Anthropic ────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # Lit ANTHROPIC_API_KEY depuis l'environnement

# ─── Prompt système (stable → mis en cache) ──────────────────────────────────
SYSTEM_PROMPT = """Tu es un agent spécialisé dans l'extraction de données de factures de transport pour Maroc Fruit Board (MFB).

Tu reçois une facture en PDF et tu dois extraire exactement les 13 champs suivants. Retourne UNIQUEMENT un objet JSON valide, sans texte autour, sans markdown, sans commentaires.

Champs à extraire :
1.  "num_facture"       — Numéro de la facture (ex: "INV-2024-001")
2.  "fournisseur"       — Nom du fournisseur / transporteur émetteur
3.  "num_commande"      — Numéro de commande ou bon de commande MFB (peut être null)
4.  "mode_transport"    — Mode : "Maritime", "Aérien" ou "Routier"
5.  "navire_vehicule"   — Nom du navire, véhicule ou numéro de vol (peut être null)
6.  "pol"               — Port/Aéroport/Ville de chargement (Point of Loading)
7.  "pod"               — Port/Aéroport/Ville de déchargement (Point of Discharge)
8.  "num_bl_cmr_lta"    — Numéro BL (maritime), CMR (routier) ou LTA (aérien) (peut être null)
9.  "nbre_unites"       — Nombre d'unités de transport (conteneurs, camions, palettes…)
10. "lieu_enlevement"   — Lieu d'enlèvement / prise en charge des marchandises
11. "date_depart"       — Date de départ au format ISO 8601 "YYYY-MM-DD" (peut être null)
12. "date_comptabilisation" — Date de comptabilisation au format ISO 8601 "YYYY-MM-DD"
13. "montant"           — Montant total de la facture en nombre décimal (ex: 15000.00)

Règles importantes :
- Si un champ est introuvable dans le document, mets null (pas de chaîne vide).
- Pour le montant, retourne uniquement le nombre, sans devise ni symbole.
- Pour les dates, convertis tout format courant (DD/MM/YYYY, MM-DD-YYYY…) en YYYY-MM-DD.
- Pour mode_transport, déduis-le du contexte si non explicitement mentionné (BL → Maritime, LTA → Aérien, CMR → Routier).
- Ne fais aucune supposition : si tu n'es pas sûr, mets null.
"""

# ─── Fonction principale d'extraction ────────────────────────────────────────
def extraire_facture(chemin_pdf: str) -> dict:
    """
    Extrait les 13 champs MFB d'une facture PDF.

    Args:
        chemin_pdf: Chemin vers le fichier PDF.

    Returns:
        Dictionnaire avec les 13 champs extraits.
    """
    pdf_path = Path(chemin_pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin_pdf}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier doit être un PDF : {chemin_pdf}")

    # Lecture et encodage base64 du PDF
    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    print(f"[MFB Agent] Traitement : {pdf_path.name} ({len(pdf_bytes) / 1024:.1f} Ko)", flush=True)

    # Appel API Claude avec document block + prompt caching
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Cache le system prompt
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

    # Extraction du texte de la réponse (ignore les blocs thinking)
    texte_reponse = ""
    for block in response.content:
        if block.type == "text":
            texte_reponse = block.text.strip()
            break

    # Nettoyage au cas où Claude encapsule dans des backticks
    if texte_reponse.startswith("```"):
        lignes = texte_reponse.split("\n")
        texte_reponse = "\n".join(lignes[1:-1]).strip()

    # Parse JSON
    try:
        resultat = json.loads(texte_reponse)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse Claude non parseable en JSON : {e}\n---\n{texte_reponse}")

    # Affichage usage tokens (utile pour surveiller les coûts)
    usage = response.usage
    print(
        f"[MFB Agent] Tokens — entrée: {usage.input_tokens} "
        f"(cache écrit: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
        f"cache lu: {getattr(usage, 'cache_read_input_tokens', 0)}) | "
        f"sortie: {usage.output_tokens}",
        flush=True,
    )

    return resultat


# ─── Validation des champs requis ────────────────────────────────────────────
CHAMPS_REQUIS = [
    "num_facture", "fournisseur", "num_commande", "mode_transport",
    "navire_vehicule", "pol", "pod", "num_bl_cmr_lta", "nbre_unites",
    "lieu_enlevement", "date_depart", "date_comptabilisation", "montant",
]

def valider_resultat(data: dict) -> dict:
    """Vérifie que tous les champs attendus sont présents (même si null)."""
    champs_manquants = [c for c in CHAMPS_REQUIS if c not in data]
    if champs_manquants:
        print(f"[MFB Agent] Avertissement — champs absents du JSON : {champs_manquants}", flush=True)
        for c in champs_manquants:
            data[c] = None
    return data


# ─── Traitement de plusieurs PDFs ────────────────────────────────────────────
def traiter_lot(chemins_pdf: list[str]) -> list[dict]:
    """Traite une liste de PDFs et retourne la liste des résultats."""
    resultats = []
    for chemin in chemins_pdf:
        try:
            data = extraire_facture(chemin)
            data = valider_resultat(data)
            data["_fichier"] = Path(chemin).name
            data["_statut"] = "ok"
            resultats.append(data)
        except Exception as e:
            resultats.append({
                "_fichier": Path(chemin).name,
                "_statut": "erreur",
                "_erreur": str(e),
            })
            print(f"[MFB Agent] ERREUR sur {chemin} : {e}", flush=True)
    return resultats


# ─── Point d'entrée CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python extract_facture.py <facture.pdf> [facture2.pdf ...]")
        sys.exit(1)

    fichiers = sys.argv[1:]

    if len(fichiers) == 1:
        # Mode simple — affiche le JSON formaté
        try:
            data = extraire_facture(fichiers[0])
            data = valider_resultat(data)
            print("\n" + json.dumps(data, ensure_ascii=False, indent=2))
        except (FileNotFoundError, ValueError) as e:
            print(f"Erreur : {e}")
            sys.exit(1)
    else:
        # Mode lot — affiche un tableau JSON
        resultats = traiter_lot(fichiers)
        print("\n" + json.dumps(resultats, ensure_ascii=False, indent=2))
        nb_ok = sum(1 for r in resultats if r.get("_statut") == "ok")
        print(f"\n[MFB Agent] {nb_ok}/{len(resultats)} factures extraites avec succès.")
