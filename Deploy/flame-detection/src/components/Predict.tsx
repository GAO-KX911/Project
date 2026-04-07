import { Card } from "antd";

export default function Predict({ title, data, hidden }: {
    title: string, data?: PredictType, hidden?: boolean
}) {
    const { inference_time, results } = data || { inference_time: 0, results: [] };

    if (hidden) {
        return <Card title={title} />
    }

    return <Card title={title}>
        <div className='w-full h-full py-4'>
            <div className="text-sm text-gray-600 mb-3">
                {inference_time > 0 && <>推理时间：{inference_time} 毫秒</>}
            </div>
            <div className="space-y-2">
                {results.map((result, index) => (
                    <div
                        key={index}
                        className="p-2 bg-blue-50 rounded text-sm"
                    >
                        {result}
                    </div>
                ))}
            </div>
        </div>
    </Card>
}
