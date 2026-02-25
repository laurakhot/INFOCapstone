// DeviceHealth is referenced in CaseData (chat.types.ts) as a global ambient
// interface so that no import statement is needed in that user-authored file.
declare interface DeviceHealth {
  signals: Array<{
    icon: string;
    label: string;
    value: string;
  }>;
}
