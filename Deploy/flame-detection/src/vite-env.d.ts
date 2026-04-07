/// <reference types="vite/client" />

type DetecRecord = {
    class: string,
    confidence: number,
    coordinates: [number, number][]
}
type DetectionType = {
    detections: DetecRecord[],
    dimensions: {
        width: number,
        height: number,
    },
    time: string,
    error?: string,
    detection_type?: 'original' | 'roi',
}

type PredictType = {
    inference_time: number,
    results: string[],
    prediction_type: 'original' | 'roi',
    error?: string,
}

type RectType = {
    x: number,
    y: number,
    width: number,
    height: number,
}

type MonitorType = "remote_camera" | 'local_camera' | "demo_video" | 'change_video'