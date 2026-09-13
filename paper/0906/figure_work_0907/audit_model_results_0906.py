"""Read-only reconciliation of the completed architecture experiment exports."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / 'outputs_0906' / 'outputs_0906'
OUT = Path(__file__).resolve().parent / 'qa'
OUT.mkdir(parents=True, exist_ok=True)
test = pd.read_csv(ROOT / 'full_test_summary/test_seed_runs.csv')
aggregate = pd.read_csv(ROOT / 'full_test_summary/test_model_summary.csv').set_index('model')
val_aggregate = pd.read_csv(ROOT / 'validation_summary/validation_model_summary.csv').set_index('model')
paired_export = pd.read_csv(ROOT / 'full_test_summary/paired_test_vs_gru.csv').set_index('left_model')
assert len(test) == 85 and test.model.nunique() == 9
assert not test.duplicated(['model', 'seed']).any()
ref_history = ref_control = ref_parents = ref_truth = None
prediction_checks = []
manifest = []
scaler_hashes = set()
for row in test.to_dict('records'):
    run = ROOT / 'development/horizon_15/lookback_60' / row['model'] / row['config_id'] / f"seed_{row['seed']}"
    m = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    tm = json.loads((run / 'full_test_evaluation/test_metrics.json').read_text(encoding='utf-8'))
    assert m['model_type'] == row['model'] and m['seed'] == row['seed']
    assert (m['lookback'], m['common_origin_lookback'], m['predict_steps']) == (60, 120, 15)
    assert m['target'] == 'Delta_Thv_15step'
    assert m['validation_parent'] == '260501' and m['held_out_test_parent'] == '0715-BACK'
    assert not m['test_data_loaded'] and not m['future_noncontrol_measurements_used'] and not m['future_label_used_as_input']
    assert m['checkpoint_selection_metric'] == 'validation_rmse_k'
    assert m['scaler_fit_source'] == 'training only after excluding 0617-ALL'
    assert m['future_control_alignment'] == 'u(t)..u(t+14)'
    assert tm['test_parent'] == '0715-BACK' and tm['test_variant'] == 'Original'
    assert not tm['training_or_weight_updates']
    assert tm['full_test_metrics']['n_windows'] == row['test_windows'] == 10659
    for key in ('test_full_rmse_k', 'test_full_mae_k', 'validation_rmse_k', 'persistence_test_rmse_k'):
        np.testing.assert_allclose(row[key], tm['summary_row'][key], rtol=0, atol=1e-10)
    np.testing.assert_allclose(row['validation_rmse_k'], m['validation_metrics']['rmse_k'], rtol=0, atol=1e-10)
    assert m['validation_windows'] == 25600 and m['train_windows'] == 306543
    if ref_history is None:
        ref_history, ref_control, ref_parents = m['history_feature_columns'], m['future_control_columns'], m['training_parents']
    assert m['history_feature_columns'] == ref_history and m['future_control_columns'] == ref_control and m['training_parents'] == ref_parents
    assert len(ref_history) == 20 and len(ref_control) == 8
    hist = pd.read_csv(run / 'training_history.csv')
    epoch_row = hist.loc[hist.epoch == m['best_epoch']].iloc[0]
    np.testing.assert_allclose(epoch_row['validation_rmse_k'], m['best_epoch_selection_metric_value_k'], rtol=0, atol=1e-10)
    # min_delta permits a sub-microkelvin gap from the literal numerical minimum.
    assert epoch_row['validation_rmse_k'] <= hist.validation_rmse_k.min() + 1.01e-6
    for name in ('metrics.json', 'best_model.pt', 'scalers.npz', 'training_history.csv', 'full_test_evaluation/test_metrics.json'):
        path = run / name
        assert path.is_file() and path.stat().st_size > 0
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({'file': path.relative_to(ROOT).as_posix(), 'sha256': digest})
        if name == 'scalers.npz':
            # NPZ zip timestamps may differ, so compare actual stored arrays below.
            with np.load(path) as values:
                scaler_hashes.add(hashlib.sha256(b''.join(k.encode() + np.asarray(values[k]).tobytes() for k in sorted(values.files))).hexdigest())
    pred_path = run / 'full_test_evaluation/test_predictions.csv'
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        assert len(pred) == 10659
        truth = pred[['frame_row_index', 'file_id', 'source_row_index', 'source_timestamp', 'current_Thv_k', 'actual_Future_Thv_15step_k']]
        if ref_truth is None:
            ref_truth = truth
        pd.testing.assert_frame_equal(truth, ref_truth)
        error = pred.predicted_Future_Thv_15step_k - pred.actual_Future_Thv_15step_k
        rmse = float(np.sqrt(np.mean(error**2)))
        np.testing.assert_allclose(rmse, row['test_full_rmse_k'], rtol=0, atol=1e-8)
        np.testing.assert_allclose(pred.predicted_Future_Thv_15step_k, pred.current_Thv_k + pred.predicted_delta_Thv_15step_k, rtol=0, atol=1e-8)
        prediction_checks.append({'model': row['model'], 'seed': row['seed'], 'n':len(pred), 'recomputed_rmse_k':rmse})
assert len(prediction_checks) == 9
assert len(scaler_hashes) == 1
base = test.loc[test.model == 'gru_baseline'].set_index('seed').test_full_rmse_k
summaries, paired = [], []
for model, data in test.groupby('model', sort=False):
    expected = {42,62,82,102,122} if model == 'tcn_baseline' else set(range(42,133,10))
    assert set(data.seed) == expected
    x = data.test_full_rmse_k.to_numpy()
    for col, source in [('test_full_rmse_k','test_full_rmse_k'),('test_full_mae_k','test_full_mae_k')]:
        np.testing.assert_allclose([data[col].mean(), data[col].std(ddof=1)], [aggregate.loc[model,source+'_mean'], aggregate.loc[model,source+'_sd']], rtol=0, atol=1e-10)
    np.testing.assert_allclose([data.validation_rmse_k.mean(),data.validation_rmse_k.std(ddof=1)], [val_aggregate.loc[model,'validation_rmse_k_mean'],val_aggregate.loc[model,'validation_rmse_k_sd']], rtol=0, atol=1e-10)
    summaries.append({'model':model,'n':len(x),'test_mean':float(x.mean()),'test_sd':float(x.std(ddof=1)),'validation_mean':float(data.validation_rmse_k.mean()),'min_seed_rmse':float(x.min()),'max_seed_rmse':float(x.max())})
    if model == 'gru_baseline': continue
    diffs = data.set_index('seed').test_full_rmse_k.subtract(base).dropna().to_numpy()
    mean = diffs.mean(); margin = stats.t.ppf(.975,len(diffs)-1)*stats.sem(diffs)
    np.testing.assert_allclose([mean,mean-margin,mean+margin],paired_export.loc[model,['mean_difference_k','ci95_low_k','ci95_high_k']].to_numpy(dtype=float),rtol=0,atol=1e-10)
    p = stats.wilcoxon(diffs, alternative='two-sided').pvalue
    np.testing.assert_allclose(p,paired_export.loc[model,'wilcoxon_p_raw'],atol=1e-10)
    paired.append({'model':model,'n':len(diffs),'mean':float(mean),'ci_low':float(mean-margin),'ci_high':float(mean+margin),'p':float(p)})
# Holm correction over all eight baseline comparisons.
raw = np.array([p['p'] for p in paired]); order = np.argsort(raw)
holm = np.empty_like(raw); holm[order] = np.minimum(1,np.maximum.accumulate(raw[order]*(len(raw)-np.arange(len(raw)))))
for p, adjusted in zip(paired,holm):
    p['holm_p'] = float(adjusted)
    np.testing.assert_allclose(adjusted,paired_export.loc[p['model'],'wilcoxon_p_holm'],atol=1e-10)
report = {'status':'PASS','completed_runs':len(test),'models':9,'saved_prediction_checks':prediction_checks,'scaler_array_signatures':len(scaler_hashes),'summary':summaries,'paired_vs_gru':paired,'limitations':['Export consistency audit, not a complete causal preprocessing or raw-data provenance audit.','Test run has already informed research choices; it is not a pristine final holdout.','Seed uncertainty is conditional on one fixed operating run.','No new training or inference performed.']}
(OUT / 'result_reconciliation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT / 'source_hash_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
