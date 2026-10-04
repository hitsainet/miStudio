/** Types for the Application Settings API. */

export type SettingCategory = 'endpoints' | 'api_keys' | 'labeling' | 'display' | 'general';

export interface AppSetting {
  id: string;
  key: string;
  value: string; // Masked if sensitive
  is_sensitive: boolean;
  category: SettingCategory;
  created_at: string;
  updated_at: string;
}

export interface AppSettingUpsert {
  key: string;
  value: string;
  is_sensitive?: boolean;
  category?: SettingCategory;
}

export interface AppSettingBulkUpsert {
  settings: AppSettingUpsert[];
}

export interface AppSettingBulkResponse {
  data: AppSetting[];
  created: number;
  updated: number;
}
