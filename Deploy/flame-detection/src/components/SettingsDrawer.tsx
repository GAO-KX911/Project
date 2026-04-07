import React from 'react';
import { Drawer, InputNumber, Form, Button, message } from 'antd';

export type SettingsDrawerItem = {
    apiInvokeIntervalRef: React.RefObject<string>;
    classifyThresholdRef: React.RefObject<string>;
    detectThresholdRef: React.RefObject<string>;
}
interface SettingsDrawerProps extends SettingsDrawerItem {
    open: boolean;
    onClose: () => void;
}

type FormValue = {
    apiInvokeInterval: string;
    classifyThreshold: string;
    detectThreshold: string;
}

const formRule = [{ required: true, message: '不可为空' }]

const SettingsDrawer: React.FC<SettingsDrawerProps> = ({
    open,
    onClose,
    apiInvokeIntervalRef,
    classifyThresholdRef,
    detectThresholdRef,
}) => {
    const onFinish = (values: FormValue) => {
        apiInvokeIntervalRef.current = values.apiInvokeInterval
        classifyThresholdRef.current = values.classifyThreshold
        detectThresholdRef.current = values.detectThreshold
        localStorage.setItem('apiInvokeInterval', values.apiInvokeInterval);
        localStorage.setItem('classifyThreshold', values.classifyThreshold);
        localStorage.setItem('detectThreshold', values.detectThreshold);
        message.success('设置成功');
        onClose();
    };

    return (
        <Drawer
            title="系统设置"
            placement="right"
            onClose={onClose}
            open={open}
            width={320}
        >
            <Form layout="vertical"
                onFinish={onFinish}
                initialValues={{
                    apiInvokeInterval: apiInvokeIntervalRef.current,
                    classifyThreshold: classifyThresholdRef.current,
                    detectThreshold: detectThresholdRef.current,
                }}
            >
                <Form.Item name="apiInvokeInterval" label="api调用频率(毫秒)" rules={formRule}>
                    <InputNumber step={100} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item name="classifyThreshold" label="识别阈值" tooltip="大于该阈值，才会进一步调用检测模型。" rules={formRule}>
                    <InputNumber step={0.1} style={{ width: '100%' }} min={0.1} max={1} />
                </Form.Item>

                <Form.Item name="detectThreshold" label="检测阈值" tooltip="大于该阈值，才会在检测结果中标出火焰位置。" rules={formRule}>
                    <InputNumber step={0.1} style={{ width: '100%' }} min={0.1} max={1} />
                </Form.Item>

                <Form.Item >
                    <Button type="primary" htmlType="submit">
                        保存
                    </Button>
                </Form.Item>
            </Form>
        </Drawer>
    );
};

export default SettingsDrawer;