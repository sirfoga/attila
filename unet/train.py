callbacks = [
    EarlyStopping(patience=10, verbose=verbose),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-5, verbose=verbose),
    ModelCheckpoint(best_model_weights, verbose=verbose, save_best_only=True, save_weights_only=True)
]
results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))

plt.figure(figsize=(24, 8))

plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(results.history["accuracy"], label="accuracy")
plt.plot(results.history["val_accuracy"], label="val_acc")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")

plt.xlabel("epochs")
plt.ylabel("log loss")
plt.legend()

# model.load_weights(best_model_weights)  # load best model
model.evaluate(X_valid, y_valid, verbose=1)  # evaluate on validation set

# predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
