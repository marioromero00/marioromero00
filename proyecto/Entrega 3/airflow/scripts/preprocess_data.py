import pandas as pd

def preprocess(df_transacciones, df_productos, df_clientes):
    """
    Preprocesa datos de transacciones, productos y clientes.
    """

    # --- 1. Convertir fechas ---
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones["purchase_date"]):
        df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])

    # --- 2. Merge con productos ---
    df_merged = df_transacciones.merge(
        df_productos,
        on="product_id",
        how="left"
    )

    # --- 3. Merge con clientes ---
    df_merged = df_merged.merge(
        df_clientes,
        on="customer_id",
        how="left"
    )

    # --- 4. Crear columnas de semana y año ---
    df_merged["week"] = df_merged["purchase_date"].dt.isocalendar().week
    df_merged["year"] = df_merged["purchase_date"].dt.isocalendar().year

    # --- 5. Crear aggregations reales ---
    agg_df = (
        df_merged
        .groupby(["customer_id", "product_id", "year", "week"], as_index=False)
        .agg(
            total_items=("items", "sum"),
            num_orders=("order_id", "nunique"),
            avg_items_per_order=("items", "mean"),
        )
    )

    # --- 6. Generar todas las combinaciones cliente-producto ---
    clientes = df_merged["customer_id"].unique()
    productos = df_merged["product_id"].unique()

    all_pairs = pd.MultiIndex.from_product(
        [clientes, productos],
        names=["customer_id", "product_id"]
    ).to_frame(index=False)

    semana_max = df_merged["week"].max()
    anio_max = df_merged.loc[df_merged["week"] == semana_max, "year"].max()

    all_pairs["week"] = semana_max
    all_pairs["year"] = anio_max

    agg_df = all_pairs.merge(
        agg_df,
        on=["customer_id", "product_id", "year", "week"],
        how="left"
    )

    agg_df.fillna(0, inplace=True)

    return agg_df
