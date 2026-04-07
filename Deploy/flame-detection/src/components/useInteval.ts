import { useEffect } from "react";

// 自定义Interval Hook
export const useInterval = (callback: () => void, delay: string, isActive: boolean, dependencies: unknown[] = []) => {
    useEffect(() => {
        let intervalId: NodeJS.Timeout;
        if (isActive) {
            intervalId = setInterval(callback, parseInt(delay));
        }
        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, [isActive, delay, ...dependencies]);
};