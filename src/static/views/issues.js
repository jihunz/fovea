// Fallback labels for issue / flag codes. Views prefer `issue_labels` from /api/datasets/{id}/stats (the server
// owns the wording); these only cover the moment before stats arrive.
export const ISSUE_LABELS = {
  missing_label: 'No label file', empty_label: 'Empty label file', bad_format: 'Malformed line', bad_class: 'Invalid class id',
  out_of_range: 'Box outside image', zero_area: 'Zero-area box', tiny_box: 'Tiny box', duplicate_box: 'Duplicate box', image_unreadable: 'Image unreadable',
  near_dup: 'Near-duplicate image', leak_split: 'Duplicate across splits', dup_same_role: 'Same frame in two folders of one split', dup_label_mismatch: 'Duplicates labeled differently',
  seq_gap: 'Box missing in one frame of a sequence', seq_flicker: 'Class flickers in a sequence', seq_jump: 'Box jumps in a sequence', seq_count_spike: 'Box count spike in a sequence',
  shape_outlier: 'Unusual box shape for its class', blurry: 'Blurry image', dark: 'Dark image', overexposed: 'Overexposed image',
  ai_missing: 'AI: likely missed object', ai_loose: 'AI: loose box', ai_swap: 'AI: possibly wrong class', ai_spurious: 'AI: box without support',
};
