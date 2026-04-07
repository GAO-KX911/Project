import { Rnd } from "react-rnd";

/**
 * 插件
 * @param param0 
 * @returns 
 */
export default function RoiArea({ roiRect, setRoiRect }: {
    roiRect: RectType,
    setRoiRect: (roiRect: RectType) => void
}) {

    return <Rnd
        bounds="parent"
        position={{ x: roiRect.x, y: roiRect.y }}  // 使用 position 替代 default
        size={{ width: roiRect.width, height: roiRect.height }}  // 使用 size 替代 default
        className="border-4 border-red-500"
        onDragStop={(_, d) => {
            setRoiRect({
                ...roiRect,
                x: d.x,
                y: d.y
            });
        }}
        onResizeStop={(_, __, ___, d, position) => {
            // console.log("roiRect d", roiRect.width, d.width);

            setRoiRect({
                x: position.x,
                y: position.y,
                width: roiRect.width + d.width,
                height: roiRect.height + d.height
            });
        }}
    >
        {/* 四个角的控制点  */}
        <div className="control-point -left-[7px] -top-[7px]"></div>
        <div className="control-point -right-[7px] -top-[7px]"></div>
        <div className="control-point -left-[7px] -bottom-[7px]"></div>
        <div className="control-point -right-[7px] -bottom-[7px]"></div>

        {/* 四条边中点的控制点 */}
        <div className="control-point -top-[6px] left-[50%]"></div>
        <div className="control-point -bottom-[7px] left-[50%]"></div>
        <div className="control-point -left-[7px] top-[50%]"></div>
        <div className="control-point -right-[7px] top-[50%]"></div>
    </Rnd>
}

