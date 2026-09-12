/** API client for application settings. */

import { fetchAPI } from './client';
import type { AppSetting, AppSettingUpsert, AppSettingBulkUpsert, AppSettingBulkResponse } from '../types/appSetting';

export async function listSettings(category?: string): Promise<AppSetting[]> {
  const query = category ? `?category=${encodeURIComponent(category)}` : '';
  return fetchAPI<AppSetting[]>(`/settings${query}`);
}

export async function getSetting(key: string): Promise<AppSetting> {
  return fetchAPI<AppSetting>(`/settings/${encodeURIComponent(key)}`);
}

export async function upsertSetting(data: AppSettingUpsert): Promise<AppSetting> {
  return fetchAPI<AppSetting>('/settings', {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function bulkUpsertSettings(data: AppSettingBulkUpsert): Promise<AppSettingBulkResponse> {
  return fetchAPI<AppSettingBulkResponse>('/settings/bulk', {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteSetting(key: string): Promise<void> {
  return fetchAPI<void>(`/settings/${encodeURIComponent(key)}`, {
    method: 'DELETE',
  });
}
