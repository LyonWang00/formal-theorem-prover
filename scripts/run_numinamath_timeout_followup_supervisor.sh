#!/usr/bin/env bash
set -euo pipefail

cd /home/lean/projects/formal-theorem-prover
campaign_root=outputs/numinamath_repair/routing_v1/campaign
queue_root=outputs/numinamath_repair/routing_v1/worker_queue
state_root=outputs/numinamath_repair/routing_v1/timeout_followup_supervisor
fail_file=lean_prover/Dataset/verified_data/numinamath_verified_fail.jsonl
index_file=outputs/numinamath_repair/routing_v1/campaign/fail_index_post_invalid.sqlite3
frozen_manifest=/mnt/e/python_project/outputs/numinamath_cloud_batches/numinamath_fail_cloud_10000.jsonl
mkdir -p "$state_root"

enqueue_if_nonempty() {
  local batch_id="$1"
  local wall_timeout="${2:-30}"
  if [[ -s "${campaign_root}/${batch_id}/candidate_manifest.jsonl" ]] \
      && [[ ! -f "${campaign_root}/${batch_id}/candidate_verification_report.json" ]] \
      && ! find "$queue_root" -mindepth 2 -maxdepth 2 -name "${batch_id}.json" -print -quit | grep -q .; then
    .venv/bin/python -m lean_prover.Dataset.numinamath_pantograph_worker enqueue \
      --queue-dir "$queue_root" \
      --batch-dir "${campaign_root}/${batch_id}" \
      --lean-project lean_project \
      --timeout "$wall_timeout"
  fi
}

while [[ ! -f "${state_root}/STOP" ]]; do
  for suffix_number in $(seq 1 29); do
    suffix=$(printf '%02d' "$suffix_number")
    source_id="zz_resource_retry120_${suffix}"
    source_batch="${campaign_root}/${source_id}"
    marker="${state_root}/${suffix}.processed"

    if [[ "$suffix" == "01" ]] && [[ -f "${campaign_root}/human_interval_120_01/prepare_report.json" ]]; then
      interval_id="human_interval_120_01"
    else
      interval_id="timeout_interval_${suffix}"
    fi
    nogoals_id="timeout_nogoals_${suffix}"

    interval_resource_id="timeout_interval_resource_${suffix}"
    if [[ -f "${campaign_root}/${interval_id}/candidate_verification_report.json" ]]; then
      if [[ ! -f "${campaign_root}/${interval_resource_id}/prepare_report.json" ]]; then
        .venv/bin/python -m lean_prover.Dataset.numinamath_repair_campaign \
          prepare-resource-followup \
          --source-batch "${campaign_root}/${interval_id}" \
          --output-root "$campaign_root" \
          --batch-id "$interval_resource_id" \
          --frozen-manifest "$frozen_manifest"
      fi
      enqueue_if_nonempty "$interval_resource_id"
    fi

    nogoals_timeout_id="zz_timeout_nogoals_${suffix}"
    if [[ -f "${campaign_root}/${nogoals_id}/candidate_verification_report.json" ]]; then
      if [[ ! -f "${campaign_root}/${nogoals_timeout_id}/prepare_report.json" ]]; then
        .venv/bin/python -m lean_prover.Dataset.numinamath_repair_campaign \
          prepare-wall-timeout-followup \
          --source-batch "${campaign_root}/${nogoals_id}" \
          --output-root "$campaign_root" \
          --batch-id "$nogoals_timeout_id" \
          --wall-timeout 120 \
          --frozen-manifest "$frozen_manifest"
      fi
      enqueue_if_nonempty "$nogoals_timeout_id" 120
    fi

    interval_resource_timeout_id="zz_timeout_interval_resource_${suffix}"
    if [[ -f "${campaign_root}/${interval_resource_id}/candidate_verification_report.json" ]]; then
      if [[ ! -f "${campaign_root}/${interval_resource_timeout_id}/prepare_report.json" ]]; then
        .venv/bin/python -m lean_prover.Dataset.numinamath_repair_campaign \
          prepare-wall-timeout-followup \
          --source-batch "${campaign_root}/${interval_resource_id}" \
          --output-root "$campaign_root" \
          --batch-id "$interval_resource_timeout_id" \
          --wall-timeout 120 \
          --frozen-manifest "$frozen_manifest"
      fi
      enqueue_if_nonempty "$interval_resource_timeout_id" 120
    fi

    if [[ -f "$marker" ]] || [[ ! -f "${source_batch}/candidate_verification_report.json" ]]; then
      continue
    fi

    if [[ ! -f "${campaign_root}/${interval_id}/prepare_report.json" ]]; then
      .venv/bin/python -m lean_prover.Dataset.numinamath_repair_campaign \
        prepare-interval-followup \
        --fail-file "$fail_file" \
        --index-file "$index_file" \
        --source-batch "$source_batch" \
        --output-root "$campaign_root" \
        --batch-id "$interval_id" \
        --frozen-manifest "$frozen_manifest"
    fi
    enqueue_if_nonempty "$interval_id"

    nogoals_edits="/mnt/e/python_project/outputs/numinamath_repair/manual_edits/${nogoals_id}.jsonl"
    if [[ ! -f "${campaign_root}/${nogoals_id}/prepare_report.json" ]]; then
      PYTHONPATH=. .venv/bin/python /mnt/e/python_project/scripts/prepare_numinamath_goal_safe_followup.py \
        --source-batch "$source_batch" \
        --edits-file "$nogoals_edits" \
        --output-root "$campaign_root" \
        --batch-id "$nogoals_id" \
        --frozen-manifest "$frozen_manifest"
    fi
    enqueue_if_nonempty "$nogoals_id"

    compatibility_id="timeout_compatibility_${suffix}"
    if [[ ! -f "${campaign_root}/${compatibility_id}/prepare_report.json" ]]; then
      .venv/bin/python -m lean_prover.Dataset.numinamath_repair_campaign \
        prepare-compatibility-followup \
        --source-batch "$source_batch" \
        --output-root "$campaign_root" \
        --batch-id "$compatibility_id" \
        --frozen-manifest "$frozen_manifest"
    fi
    enqueue_if_nonempty "$compatibility_id"

    printf '%s\n' "$(date -u +%FT%TZ) ${source_id}" > "$marker"
  done

  prepared_timeout_batches=$(find "$campaign_root" -maxdepth 2 -path '*/zz_resource_retry120_*/prepare_report.json' | wc -l)
  priority_jobs=$(find "$queue_root/pending" "$queue_root/running" -maxdepth 1 -type f \
    \( -name 'zz_*.json' -o -name 'timeout_*.json' -o -name 'human_*.json' \) | wc -l)
  if [[ "$prepared_timeout_batches" -eq 29 ]] && [[ "$priority_jobs" -eq 0 ]]; then
    if [[ ! -f "${state_root}/local_pilot_enqueued" ]]; then
      enqueue_if_nonempty fs_local_untried_0001
      enqueue_if_nonempty fs_local_untried_0002
      printf '%s\n' "$(date -u +%FT%TZ)" > "${state_root}/local_pilot_enqueued"
    elif [[ ! -f "${state_root}/local_expansion_decided" ]] \
        && [[ -f "${campaign_root}/fs_local_untried_0001/candidate_verification_report.json" ]] \
        && [[ -f "${campaign_root}/fs_local_untried_0002/candidate_verification_report.json" ]]; then
      local_solved=$(grep -h '"records_solved"' \
        "${campaign_root}/fs_local_untried_0001/candidate_verification_report.json" \
        "${campaign_root}/fs_local_untried_0002/candidate_verification_report.json" \
        | tr -cd '0-9\n' | awk '{sum += $1} END {print sum + 0}')
      if (( local_solved >= 3 )); then
        for local_number in $(seq 3 8); do
          local_suffix=$(printf '%04d' "$local_number")
          enqueue_if_nonempty "fs_local_untried_${local_suffix}"
        done
        printf '%s\n' "expand solved=${local_solved}" > "${state_root}/local_expansion_decided"
      else
        printf '%s\n' "stop_low_yield solved=${local_solved}" > "${state_root}/local_expansion_decided"
      fi
    fi
  fi
  sleep 5
done
