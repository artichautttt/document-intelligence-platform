# Plan d'hébergement de l'API sur AWS (non déployé)

**État : plan seulement. Aucune ressource AWS n'a été créée.** Ce document
décrit ce qui serait déployé, son coût estimé, l'alerte de budget, la procédure
d'arrêt complet et les risques. Le déploiement attend une validation explicite.

## Point bloquant actuel

Le crédit de l'API Anthropic est épuisé et ne sera pas rechargé pour l'instant.
Une API déployée aujourd'hui servirait seulement les fonctions sans LLM :
`/health`, `/docs`, l'ingestion (`POST /documents`) et les métriques. `POST /query`
renverrait `503` (comportement vérifié sans clé, voir le README). Une démo
publique n'a donc d'intérêt qu'une fois le crédit rechargé. D'ici là, il est
recommandé de ne pas déployer.

## Architecture proposée : une instance EC2, Docker Compose réduit

| Élément | Choix | Raison |
|---|---|---|
| Région | `eu-west-3` (Paris) | proximité, données sans caractère personnel |
| Calcul | EC2 `t3.small` (2 vCPU, 2 Go), Amazon Linux 2023 | l'image API pèse 1,64 Go (mesuré, voir *Docker*) et embarque ChromaDB, `unstructured` et un modèle ONNX : 1 Go de RAM (`t3.micro`) est vraisemblablement trop juste, non mesuré |
| Stockage | EBS gp3 20 Go, chiffré | image Docker + volume ChromaDB |
| Services | seulement `api` (ChromaDB embarqué) | Prometheus, Grafana, Qdrant et MLflow restent locaux : moins de RAM, moins de surface exposée |
| Entrée | port 443 via Caddy (HTTPS automatique) devant `api:8000` ; port 8000 non exposé | chiffrement de la clé `X-API-Key` en transit |
| Administration | AWS Systems Manager Session Manager, aucun port SSH ouvert | pas de clé SSH à protéger |
| Secrets | SSM Parameter Store (`SecureString`) pour `API_KEY` et `ANTHROPIC_API_KEY`, lus au démarrage par un rôle IAM d'instance | aucun secret dans le dépôt, l'image ou le user-data |
| Étiquetage | toutes les ressources avec `Project=document-intelligence` | inventaire et suppression complète vérifiables |

Alternatives écartées : App Runner et ECS Fargate (facturation à la ressource
allouée, plus chère pour un service allumé en permanence, et plus de pièces à
configurer) ; Lightsail containers (offre la plus petite trouvée à 1 Go de RAM,
prix non confirmés sur une page officielle au moment de la rédaction).

## Coût mensuel estimé (à confirmer dans le AWS Pricing Calculator le jour J)

| Poste | Estimation | Source |
|---|---|---|
| EC2 `t3.small` à la demande, Paris, 730 h | ≈ 17,2 $ (0,0236 $/h) | comparateurs tiers (Holori, DoiT), janvier 2026 ou plus récent |
| EBS gp3 20 Go | ≈ 2 $ | ordre de grandeur, non vérifié |
| Adresse IPv4 publique | ≈ 3,6 $ (0,005 $/h) | tarif public annoncé par AWS en 2024, non revérifié |
| Transfert sortant | négligeable pour une démo | – |
| SSM, Parameter Store standard, Budgets | 0 $ dans les limites d'usage gratuites | à vérifier |
| **Total** | **≈ 23 $/mois**, couvert par les crédits Free Tier du compte (valables jusqu'en mars 2027 selon le propriétaire du compte) | – |

Ces montants sont des estimations : aucune facture réelle n'a été observée. Les
crédits ne couvrent pas forcément tous les services ; à vérifier dans la console
(Billing → Credits).

Coût Anthropic séparé : chaque `/query` coûte de l'ordre de 0,006 $ avec
`claude-sonnet-5` (un appel mesuré : 2 166 tokens en entrée, 201 en sortie,
0,0063 $). Un usage abusif de la démo se paierait sur ce compte, d'où les
protections ci-dessous.

## Alerte de budget (à créer avant toute autre ressource)

1. **AWS Budgets** : budget mensuel de coût de 10 $, **crédits exclus du calcul**
   (option « Include credits » décochée), pour voir la consommation réelle même
   quand les crédits la paient. Alertes e-mail à 50 %, 80 % et 100 % du réel, et
   à 100 % du prévisionnel.
2. **Cost Anomaly Detection** : moniteur par service, alerte e-mail.
3. **Anthropic** : limite de dépense mensuelle basse sur la clé utilisée par la
   démo (console Anthropic, limites de workspace), et clé dédiée à la démo.

## Procédure d'arrêt complet

Toutes les ressources portent `Project=document-intelligence`. Dans l'ordre :

```bash
REGION=eu-west-3
# 1. Instance (supprime aussi le volume racine si DeleteOnTermination=true)
aws ec2 terminate-instances --region $REGION --instance-ids <instance-id>
aws ec2 wait instance-terminated --region $REGION --instance-ids <instance-id>
# 2. Adresse IP élastique éventuelle
aws ec2 release-address --region $REGION --allocation-id <eipalloc-id>
# 3. Volumes et instantanés restants
aws ec2 describe-volumes --region $REGION --filters Name=tag:Project,Values=document-intelligence
aws ec2 delete-volume --region $REGION --volume-id <vol-id>
aws ec2 describe-snapshots --region $REGION --owner-ids self --filters Name=tag:Project,Values=document-intelligence
# 4. Groupe de sécurité, rôle IAM et profil d'instance, paramètres SSM
aws ec2 delete-security-group --region $REGION --group-id <sg-id>
aws ssm delete-parameters --region $REGION --names /document-intelligence/API_KEY /document-intelligence/ANTHROPIC_API_KEY
aws iam remove-role-from-instance-profile --instance-profile-name document-intelligence --role-name document-intelligence
aws iam delete-instance-profile --instance-profile-name document-intelligence
aws iam detach-role-policy --role-name document-intelligence --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role-policy --role-name document-intelligence --policy-name read-ssm-params
aws iam delete-role --role-name document-intelligence
# 5. Vérification : plus aucune ressource étiquetée
aws resourcegroupstaggingapi get-resources --region $REGION --tag-filters Key=Project,Values=document-intelligence
```

Puis, côté Anthropic, révoquer la clé dédiée à la démo. Garder l'alerte de
budget un mois de plus pour confirmer une facture nulle, puis la supprimer.

## Risques

| Risque | Mesure |
|---|---|
| **Accès public** : n'importe qui peut appeler l'API | `API_KEY` obligatoire (`X-API-Key` sur `/documents`, `/query`, `/metrics`, déjà implémenté) ; `/health` reste public. Pas de limitation de débit dans le code : à ajouter (Caddy ou middleware) avant d'ouvrir largement |
| **Fuite de la clé Anthropic** | clé dédiée, limite de dépense basse, stockée dans SSM, jamais dans l'image, le dépôt ou les logs ; révocation dans la procédure d'arrêt |
| **Coût AWS qui dérive** (instance oubliée, IP ou volumes orphelins) | alerte de budget hors crédits, étiquetage systématique, vérification finale par tag |
| **Coût Anthropic par abus** | `API_KEY`, limite de dépense, `/query` désactivable en retirant la clé |
| **Documents déposés par des tiers** | ingestion protégée par `API_KEY` ; corpus de démo limité aux documents de test ; garde-fous activables (`GUARDRAILS_ENABLED`), avec leurs limites mesurées (README, *Garde-fous*) |
| **Données perdues** au redémarrage | ChromaDB sur volume EBS ; corpus de démo réindexable depuis `tests/fixtures/documents/` |
| **Image non mise à jour** (failles) | reconstruction depuis `main` à chaque déploiement ; démo à durée limitée, arrêtée après usage |

## Ce qui serait mesuré après un déploiement validé

Un aller-retour réel depuis l'extérieur (`/health`, upload d'un document de
test, une `/query` si le crédit le permet), avec latence et coût rapportés tels
quels, puis l'URL, l'état et la procédure de suppression ajoutés au README.
