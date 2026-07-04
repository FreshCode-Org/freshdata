"""Shared fixtures for the Phase-4 learning-profile test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.learning import learn


def make_orders_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The spec's e-commerce paired example (messy, clean)."""
    messy = pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "email": [
                " Asha@GMAIL.COM ",
                "bob@@yahoo.com",
                "carol @ hotmail.com",
                "dev@site.com",
                "eve@example.com",
                "frank@@mail.com",
                "gia@@web.com",
                "hank@@zap.com",
            ],
            "phone": [
                "98765 43210",
                "9876543211",
                "09876543212",
                "98765-43213",
                "9876543214",
                "98765 43215",
                "9876543216",
                "98765 43217",
            ],
            "status": [
                "SHIPPED",
                "shipped ",
                "Deliverd",
                "delivered",
                "SHIPPED",
                "Deliverd",
                "shipped ",
                "SHIPPED",
            ],
            "order_date": [
                "02/01/2024",
                "03/01/2024",
                "15/01/2024",
                "20/01/2024",
                "21/01/2024",
                "25/01/2024",
                "28/01/2024",
                "30/01/2024",
            ],
            "amount": [
                "₹1,200",
                "₹950",
                "₹2,100",
                "₹700",
                "₹1,500",
                "₹800",
                "₹999",
                "₹1,050",
            ],
        }
    )
    clean_df = pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "email": [
                "asha@gmail.com",
                "bob@yahoo.com",
                "carol@hotmail.com",
                "dev@site.com",
                "eve@example.com",
                "frank@mail.com",
                "gia@web.com",
                "hank@zap.com",
            ],
            "phone": [
                "+919876543210",
                "+919876543211",
                "+919876543212",
                "+919876543213",
                "+919876543214",
                "+919876543215",
                "+919876543216",
                "+919876543217",
            ],
            "status": [
                "shipped",
                "shipped",
                "delivered",
                "delivered",
                "shipped",
                "delivered",
                "shipped",
                "shipped",
            ],
            "order_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-15",
                    "2024-01-20",
                    "2024-01-21",
                    "2024-01-25",
                    "2024-01-28",
                    "2024-01-30",
                ]
            ),
            "amount": [1200.0, 950.0, 2100.0, 700.0, 1500.0, 800.0, 999.0, 1050.0],
        }
    )
    return messy, clean_df


def make_new_batch() -> pd.DataFrame:
    """A fresh batch with the same corruption families as the training pair."""
    return pd.DataFrame(
        {
            "order_id": ["B1", "B2", "B3"],
            "email": ["ivan@@gmail.com", " Judy@YAHOO.COM ", "ken@site.com"],
            "phone": ["98765 99990", "9876599991", "98765-99992"],
            "status": ["Deliverd", "SHIPPED", "shipped "],
            "order_date": ["05/02/2024", "07/02/2024", "09/02/2024"],
            "amount": ["₹500", "₹1,000", "₹750"],
        }
    )


@pytest.fixture(scope="session")
def orders_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_orders_pair()


@pytest.fixture(scope="session")
def orders_profile(orders_pair):
    messy, clean_df = orders_pair
    return learn(
        messy,
        clean_df,
        context="order_id is a unique identifier. Never modify order_id.",
        key="order_id",
        dataset_id="orders",
        min_support=2,
    )


@pytest.fixture()
def new_batch() -> pd.DataFrame:
    return make_new_batch()
