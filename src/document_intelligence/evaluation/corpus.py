"""Corpus synthétique étendu pour l'évaluation : trois documents de 10 à 12 sections.

Les documents sont **fictifs** (sociétés, montants et dates inventés) et générés
de façon déterministe au format DOCX, avec titres structurés et un tableau
chacun. Ils complètent les fixtures de `tests/fixtures/documents/`, trop courtes
pour que le retrieval ait quelque chose à départager.
"""

from pathlib import Path

from docx import Document

# Chaque document : (nom de fichier, titre, [(section, [paragraphes])], tableau)
_Table = list[list[str]]
_Section = tuple[str, list[str]]

_LICENCE_CONTRACT: tuple[str, str, list[_Section], _Table] = (
    "contrat_licence_helio.docx",
    "Contrat de licence et de maintenance logicielle",
    [
        (
            "Préambule",
            [
                "Le présent contrat est conclu entre la société Helio Analytics SAS, "
                "éditeur de la plateforme HelioCore, et le Groupe Verdier SA, ci-après "
                "le client."
            ],
        ),
        (
            "Article 1 - Objet",
            [
                "L'éditeur concède au client une licence non exclusive et non "
                "transférable d'utilisation de la plateforme HelioCore, pour un maximum "
                "de 250 utilisateurs nommés."
            ],
        ),
        (
            "Article 2 - Durée",
            [
                "Le contrat prend effet le 1er mars 2026 pour une durée de trois ans. "
                "Il se renouvelle tacitement par périodes d'un an, sauf notification de "
                "non-renouvellement adressée avec un préavis de quatre-vingt-dix jours."
            ],
        ),
        (
            "Article 3 - Redevances",
            [
                "La redevance annuelle s'élève à 184 000 euros HT, facturée par "
                "échéances trimestrielles de 46 000 euros HT. Elle est révisée chaque "
                "année selon l'indice Syntec, avec une hausse plafonnée à 4 pour cent."
            ],
        ),
        (
            "Article 4 - Niveaux de service",
            [
                "L'éditeur garantit une disponibilité mensuelle de 99,7 pour cent, hors "
                "opérations de maintenance planifiée notifiées cinq jours à l'avance."
            ],
        ),
        (
            "Article 5 - Pénalités de service",
            [
                "Lorsque la disponibilité mensuelle est inférieure à l'engagement, le "
                "client obtient un crédit de 5 pour cent de la redevance trimestrielle "
                "par tranche de 0,5 point de disponibilité manquant, dans la limite de "
                "25 pour cent de cette redevance."
            ],
        ),
        (
            "Article 6 - Support",
            [
                "Le support est joignable par téléphone et par courriel les jours "
                "ouvrés de 8 heures à 19 heures. Un incident critique est pris en "
                "charge en deux heures et fait l'objet d'une résolution ou d'un "
                "contournement sous huit heures."
            ],
        ),
        (
            "Article 7 - Propriété intellectuelle",
            [
                "L'éditeur conserve l'intégralité des droits sur la plateforme et ses "
                "évolutions. Le client reste seul propriétaire des données qu'il y "
                "importe et peut en demander l'export à tout moment."
            ],
        ),
        (
            "Article 8 - Protection des données",
            [
                "Les données du client sont hébergées exclusivement dans l'Union "
                "européenne, au sein du centre de données de Francfort. L'éditeur "
                "notifie toute violation de données personnelles au client dans un "
                "délai de 48 heures après en avoir eu connaissance."
            ],
        ),
        (
            "Article 9 - Confidentialité",
            [
                "Chaque partie s'engage à garder confidentielles les informations "
                "reçues de l'autre, pendant la durée du contrat et pendant sept ans "
                "après son terme."
            ],
        ),
        (
            "Article 10 - Responsabilité",
            [
                "La responsabilité de l'éditeur est plafonnée au montant des "
                "redevances versées au cours des douze mois précédant le fait "
                "générateur. Les dommages indirects, tels que la perte de chiffre "
                "d'affaires, sont exclus."
            ],
        ),
        (
            "Article 11 - Résiliation",
            [
                "Chaque partie peut résilier le contrat en cas de manquement non "
                "corrigé dans un délai de 45 jours après mise en demeure. Le client "
                "peut en outre résilier pour convenance à compter du vingt-cinquième "
                "mois, moyennant un préavis de six mois."
            ],
        ),
        (
            "Article 12 - Droit applicable",
            [
                "Le contrat est soumis au droit français. Tout litige relève de la "
                "compétence exclusive du tribunal de commerce de Lyon."
            ],
        ),
        (
            "Annexe - Tarifs des prestations additionnelles",
            ["Les prestations hors redevance sont facturées selon le barème suivant."],
        ),
    ],
    [
        ["Prestation", "Tarif (EUR)"],
        ["Formation (par jour)", "1 200"],
        ["Migration de données (forfait)", "18 500"],
        ["Utilisateur supplémentaire (par an)", "640"],
    ],
)

_HALF_YEAR_REPORT: tuple[str, str, list[_Section], _Table] = (
    "rapport_semestriel_nordvale.docx",
    "Rapport financier semestriel - Nordvale Industries - S1 2026",
    [
        (
            "Faits marquants",
            [
                "Le chiffre d'affaires du premier semestre 2026 atteint 63,4 millions "
                "d'euros, en croissance de 8,7 pour cent. Le groupe a finalisé "
                "l'acquisition de la société Tecnomar pour 14,2 millions d'euros."
            ],
        ),
        (
            "Résultats du semestre",
            [
                "L'EBITDA ressort à 11,9 millions d'euros, soit une marge d'EBITDA de "
                "18,8 pour cent. Le résultat net part du groupe s'établit à 6,1 "
                "millions d'euros, soit 0,92 euro par action."
            ],
        ),
        (
            "Activité par segment",
            [
                "Le segment Industrie représente 39,1 millions d'euros de chiffre "
                "d'affaires, et le segment Services 24,3 millions d'euros. Le segment "
                "Services progresse plus vite grâce aux contrats de maintenance "
                "récurrents."
            ],
        ),
        (
            "Endettement",
            [
                "La dette financière nette s'élève à 27,5 millions d'euros au 30 juin "
                "2026, ce qui représente 1,2 fois l'EBITDA annualisé."
            ],
        ),
        (
            "Trésorerie et investissements",
            [
                "Le flux de trésorerie opérationnel atteint 9,3 millions d'euros. Les "
                "investissements industriels du semestre s'élèvent à 4,8 millions "
                "d'euros. La trésorerie disponible est de 21,6 millions d'euros."
            ],
        ),
        (
            "Ressources humaines",
            [
                "Le groupe emploie 1 240 collaborateurs, dont 38 pour cent hors de "
                "France. Le turnover volontaire reste inférieur à 7 pour cent."
            ],
        ),
        (
            "Politique de dividende",
            [
                "Le conseil d'administration a décidé le versement d'un acompte sur "
                "dividende de 0,30 euro par action, mis en paiement en septembre 2026."
            ],
        ),
        (
            "Facteurs de risque",
            [
                "Trois clients représentent 31 pour cent du chiffre d'affaires du "
                "groupe, ce qui crée une dépendance commerciale. Le groupe est aussi "
                "exposé à la livre sterling et à la hausse du coût de l'énergie."
            ],
        ),
        (
            "Perspectives",
            [
                "Le groupe vise une croissance de son chiffre d'affaires annuel "
                "comprise entre 9 et 11 pour cent, avec une marge d'EBITDA supérieure "
                "à 18 pour cent."
            ],
        ),
        (
            "Événements postérieurs à la clôture",
            [
                "Un contrat-cadre de cinq ans a été signé en juillet 2026 avec le "
                "client Kessler Rail pour la fourniture de sous-ensembles mécaniques."
            ],
        ),
        (
            "Détail trimestriel du chiffre d'affaires",
            ["Le chiffre d'affaires se répartit comme suit entre les deux trimestres."],
        ),
    ],
    [
        ["Trimestre", "Chiffre d'affaires (M EUR)"],
        ["T1 2026", "30,2"],
        ["T2 2026", "33,2"],
    ],
)

_SECURITY_AUDIT: tuple[str, str, list[_Section], _Table] = (
    "audit_securite_si.docx",
    "Rapport d'audit de sécurité du système d'information",
    [
        (
            "Contexte et périmètre",
            [
                "La mission s'est déroulée du 4 au 29 mai 2026 sur trois applications "
                "critiques, 120 serveurs et deux sites de l'entreprise."
            ],
        ),
        (
            "Méthodologie",
            [
                "L'audit s'appuie sur l'annexe A de la norme ISO 27001 et sur les "
                "recommandations de l'ANSSI. Il combine des tests d'intrusion et des "
                "entretiens avec 14 responsables métiers et techniques."
            ],
        ),
        (
            "Gestion des accès",
            [
                "Nous avons identifié 17 comptes administrateurs orphelins, non "
                "rattachés à un collaborateur en poste. L'authentification "
                "multifacteur est absente de l'accès VPN des prestataires externes."
            ],
        ),
        (
            "Gestion des correctifs",
            [
                "Sur l'ensemble du parc, 23 pour cent des serveurs présentent un "
                "retard de plus de 60 jours dans l'application des correctifs "
                "critiques."
            ],
        ),
        (
            "Sauvegardes et continuité",
            [
                "Les sauvegardes sont quotidiennes, mais aucun test de restauration "
                "n'a jamais été réalisé sur l'application de paie. L'objectif de "
                "point de reprise est de 24 heures et l'objectif de temps de reprise "
                "de 12 heures, valeurs théoriques non éprouvées."
            ],
        ),
        (
            "Journalisation",
            [
                "Les journaux d'événements sont conservés 90 jours, alors que la "
                "politique de sécurité de l'entreprise exige une conservation de "
                "12 mois."
            ],
        ),
        (
            "Sensibilisation des collaborateurs",
            [
                "Lors de la campagne d'hameçonnage simulé, 26 pour cent des "
                "collaborateurs ont cliqué sur le lien piégé et 9 pour cent ont saisi "
                "leurs identifiants."
            ],
        ),
        (
            "Sécurité physique",
            [
                "La salle serveur du site de Nantes n'est pas protégée par un "
                "contrôle d'accès par badge : la porte se ferme avec une simple clé."
            ],
        ),
        (
            "Synthèse des risques",
            [
                "Au total, 15 vulnérabilités ont été relevées : 2 risques critiques, "
                "5 risques majeurs et 8 risques modérés."
            ],
        ),
        (
            "Recommandations",
            [
                "Il est recommandé de déployer l'authentification multifacteur sur "
                "tous les accès distants avant le 30 septembre 2026, de supprimer les "
                "comptes orphelins sous 30 jours, de porter la rétention des journaux "
                "à 12 mois et de réaliser un test de restauration chaque trimestre."
            ],
        ),
        (
            "Plan d'action",
            [
                "Le pilotage du plan d'action est confié au responsable de la "
                "sécurité des systèmes d'information. Le budget estimé est de "
                "220 000 euros."
            ],
        ),
        (
            "Répartition des vulnérabilités par sévérité",
            [
                "Le tableau ci-dessous détaille les vulnérabilités par niveau de sévérité."
            ],
        ),
    ],
    [
        ["Sévérité", "Nombre"],
        ["Critique", "2"],
        ["Majeur", "5"],
        ["Modéré", "8"],
    ],
)

_DOCUMENTS = [_LICENCE_CONTRACT, _HALF_YEAR_REPORT, _SECURITY_AUDIT]


def build_extended_corpus(directory: Path) -> list[Path]:
    """Écrit les documents du corpus étendu dans `directory` et retourne leurs chemins."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, title, sections, table_rows in _DOCUMENTS:
        doc = Document()
        doc.add_heading(title, level=0)
        for heading, paragraphs in sections:
            doc.add_heading(heading, level=1)
            for paragraph in paragraphs:
                doc.add_paragraph(paragraph)
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
        path = directory / filename
        doc.save(str(path))
        paths.append(path)
    return paths
