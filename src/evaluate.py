from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_output_model(predictions: object, actual: object, metrics: object = ['MSE', 'MAE', 'R2']) -> object:
    """
    Évalue les performances d'un modèle multi-sorties en utilisant différentes métriques.

    :param predictions: Les prédictions du modèle. Attendu d'avoir une forme [n_samples, n_outputs].
    :param actual: Les valeurs réelles. Doit avoir la même forme que les prédictions.
    :param metrics: Liste des métriques d'évaluation à utiliser. Par défaut : MSE, MAE et R².
    :return: Un dictionnaire de dictionnaires contenant les scores pour chaque métrique spécifiée pour chaque cible.
    """
    results = {}
    if 'MSE' in metrics:
        results['MSE'] = mean_squared_error(actual, predictions)
    if 'MAE' in metrics:
        results['MAE'] = mean_absolute_error(actual, predictions)
    if 'R2' in metrics:
        results['R2'] = r2_score(actual, predictions)

    return results

def print_evaluation_results(results):
    """
    Affiche les résultats d'évaluation pour un modèle multi-sorties.

    :param results: Dictionnaire contenant les résultats d'évaluation pour chaque cible.
    """
    print("Résultats de l'évaluation :")
    for metric, score in results.items():
        print(f"  {metric}: {score}")