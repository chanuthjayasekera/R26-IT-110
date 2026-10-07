import Layer from "express/lib/router/layer.js";

function wrapHandler(handler) {
  if (typeof handler !== "function" || handler.length === 4) {
    return handler;
  }

  return function asyncErrorHandler(req, res, next) {
    const result = handler.call(this, req, res, next);
    if (result && typeof result.catch === "function") {
      result.catch(next);
    }
    return result;
  };
}

const descriptor = Object.getOwnPropertyDescriptor(Layer.prototype, "handle");

if (!descriptor || descriptor.configurable) {
  Object.defineProperty(Layer.prototype, "handle", {
    configurable: true,
    enumerable: true,
    get() {
      return this.__handle;
    },
    set(handler) {
      this.__handle = wrapHandler(handler);
    }
  });
}
