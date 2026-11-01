import React from "react";
import { Composition } from "remotion";
import { KnobTuning } from "./KnobTuning";

export const RemotionRoot: React.FC = () => (
  <Composition
    id="KnobTuning"
    component={KnobTuning}
    durationInFrames={210}
    fps={30}
    width={960}
    height={540}
    defaultProps={{}}
  />
);
