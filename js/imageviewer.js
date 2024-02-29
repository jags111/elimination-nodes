// import { app } from "../../scripts/app.js";
// import { ComfyWidgets } from "../../scripts/widgets.js";

// // Adds an upload button to the nodes

// const nodesWithUploadFields = ["base_image", "alpha_overlay"];

// console.log("Adding upload buttons to nodes");
// app.registerExtension({
//   name: "CompositeAlphatoBase",
//   async beforeRegisterNodeDef(nodeType, nodeData, app) {
//     if (nodeData.name !== "CompositeAlphatoBase") {
//         return;
//         }
//     for (const uploadField of nodesWithUploadFields) {
//       if (
//         nodeData?.input?.required &&
//         nodeData.input.required[uploadField]?.[1]?.image_upload === true
//       ) {
//         console.log("Adding upload button to", nodeType);
//         console.log("nodeType", nodeType);
//         console.log("nodeData", nodeData);
//         console.log("app", app);
//         nodeData.input.required.upload = ["IMAGEUPLOAD"];
//       }
//     }
//   },
// });
